import torch
from x_transformers.autoregressive_wrapper import top_k

from midi_lm import logger

# start of song
# one instrument (piano)
# start of notes
SOS_SEED = {
    "event_type": torch.tensor([[0, 1, 2]], dtype=torch.long),
    "beat": torch.tensor([[0, 0, 0]], dtype=torch.long),
    "position": torch.tensor([[0, 0, 0]], dtype=torch.long),
    "pitch": torch.tensor([[0, 0, 0]], dtype=torch.long),
    "duration": torch.tensor([[0, 0, 0]], dtype=torch.long),
    "instrument": torch.tensor([[0, 1, 0]], dtype=torch.long),
    "attention_mask": torch.tensor([[1, 1, 1]], dtype=torch.bool),
}


# once instruments are selected, you can only sample those instruments
@torch.no_grad()
def generate_from_seed(
    model,
    input_seed: dict[str, torch.Tensor] = SOS_SEED,
    temperature: float = 1.0,
    top_k_fraction: float = 0.9,
    min_steps: int = 1,
    steps: int = 1,
    device: str | torch.device = "cpu",
    monotonic_keys=("event_type", "beat"),
):
    logger.info("generating sequence")
    assert min_steps <= steps, "min_steps must be less than or equal to steps"

    device = torch.device(device)
    generated_sequence = {}
    for k, v in input_seed.items():
        generated_sequence[k] = v.clone().detach().to(device)

    current_type = generated_sequence["event_type"][:, -1]

    allowed_instruments = torch.tensor([], dtype=torch.long).to(device)
    if current_type >= 1:
        # read the instruments provided from the seed
        # NOTE: when i refactor this to work better for batch generation,
        # the dimension will need to be set here but the operation isn't supported yet for MPS devices
        # i'll wait to implement this until i need it
        allowed_instruments = torch.cat([allowed_instruments, generated_sequence["instrument"].unique()])

    model = model.to(device)

    # input_seq shape: (batch_dim, seq_len)
    for step in range(steps):
        # for specified keys, mask out tokens less than the current max value
        monotocity_thresholds = {
            k: v.max().item() for k, v in generated_sequence.items() if k in monotonic_keys
        }

        output = model(generated_sequence)
        for key in ["event_type", "beat", "position", "pitch", "duration", "instrument"]:
            logits = output[key]
            # logits shape: (batch_size, seq_len, vocab_size)
            # select logits at final step
            next_token_logits = logits[:, -1, :]
            # next_token_logits shape: (batch_size, vocab_size)

            # apply various masks
            if key in monotocity_thresholds:
                # mask out tokens with index less than monotocity_threshold
                next_token_logits[:, : monotocity_thresholds[key]] = -float("inf")

            if current_type == 3 and key in ("beat", "position", "pitch", "duration", "instrument"):
                # mask out the padding index
                next_token_logits[:, 0] = -float("inf")

            if current_type == 3 and key == "instrument":
                # mask out all instruments that aren't in the allowed instruments
                vocab_size = next_token_logits.shape[1]
                mask = torch.ones(vocab_size, dtype=torch.bool).to(device)
                mask[allowed_instruments] = False
                next_token_logits.masked_fill_(mask, -float("inf"))

            if steps <= min_steps and key == "event_type":
                # mask out the end of sequence token
                next_token_logits[:, 4] = -float("inf")

            # apply temperate and top_k filtering
            next_token_logits = top_k(next_token_logits / temperature, frac_num_tokens=top_k_fraction)

            # normalize scaled logits to get predicted probabilities
            sample_distribution = torch.softmax(next_token_logits, dim=1)
            # sample from distribution to get predicted output token
            next_input = torch.multinomial(sample_distribution, num_samples=1)
            # next_input shape: (batch_size, num_samples)
            # add the output to the context window that we repeatedly feed the model
            generated_sequence[key] = torch.concat((generated_sequence[key], next_input), dim=1)

            # keep track of what type of event we're generating
            if key == "event_type":
                current_type = generated_sequence["event_type"][:, -1]

            if key == "instrument" and current_type == 2:
                allowed_instruments = torch.cat([allowed_instruments, next_input.squeeze(dim=1)])

        generated_sequence["attention_mask"] = torch.cat(
            (generated_sequence["attention_mask"], torch.tensor([[True]]).to(device)), dim=1
        )

        # check if we predicted the end of sequence token
        if generated_sequence["event_type"][:, -1] == 4:
            logger.info(f"generated {step} steps before predicting end of sequence")
            break

    return generated_sequence
