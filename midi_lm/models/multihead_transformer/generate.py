import torch
from x_transformers.autoregressive_wrapper import top_k

from midi_lm import logger

BOS_SEED = {
    "time_shift": torch.tensor([[0]], dtype=torch.long),
    "pitch": torch.tensor([[1]], dtype=torch.long),
    "duration": torch.tensor([[0]], dtype=torch.long),
    "attention_mask": torch.tensor([[1]], dtype=torch.bool),
}


def generate_from_seed(
    model,
    eos_key,
    eos_token_id,
    input_seed: dict[str, torch.Tensor] = BOS_SEED,
    temperature: float = 1.0,
    top_k_fraction: float = 0.9,
    min_steps: int = 1,
    steps: int = 1,
    device: str | torch.device = "cpu",
):
    logger.info("generating sequence")
    assert min_steps <= steps, "min_steps must be less than or equal to steps"

    device = torch.device(device)
    generated_sequence = {}
    for k, v in input_seed.items():
        generated_sequence[k] = v.clone().detach().to(device)

    model = model.to(device)

    # input_seq shape: (batch_dim, seq_len)
    for step in range(steps):
        output = model(generated_sequence)
        for key in model.config.vocab_sizes.keys():
            logits = output[key]
            # logits shape: (batch_size, seq_len, vocab_size)
            next_token_logits = logits[:, -1, :]
            if step <= min_steps and key == eos_key:
                # if we haven't generated the minimum number of steps, mask out the eos token
                next_token_logits[:, eos_token_id] = -float("inf")

            # next_token_logits shape: (batch_size, vocab_size)
            next_token_logits = top_k(next_token_logits / temperature, frac_num_tokens=top_k_fraction)
            # normalize scaled logits to get predicted probabilities
            sample_distribution = torch.softmax(next_token_logits, dim=1)
            # sample from distribution to get predicted output token
            next_input = torch.multinomial(sample_distribution, num_samples=1)
            # next_input shape: (batch_size, num_samples)
            # add the output to the context window that we repeatedly feed the model
            generated_sequence[key] = torch.concat((generated_sequence[key], next_input), dim=1)

        generated_sequence["attention_mask"] = torch.cat(
            (generated_sequence["attention_mask"], torch.tensor([[True]]).to(device)), dim=1
        )

        # check if we predicted the end of sequence token
        if generated_sequence[eos_key][:, -1] == eos_token_id:
            logger.info(f"generated {step} steps before predicting end of sequence")
            break
    return generated_sequence
