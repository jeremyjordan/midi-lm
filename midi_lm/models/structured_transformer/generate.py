import torch
from x_transformers.autoregressive_wrapper import top_k

from midi_lm import logger
from midi_lm.models.structured_transformer.network import StructuredTransformer

BOS_SEED = {"token_ids": torch.tensor([[1]]), "attention_mask": torch.tensor([[True]])}

TOKEN_RANGES = {
    "special": (0, 2),
    "pitch": (3, 130),
    "velocity": (131, 162),
    "duration": (163, 418),
    "timeshift": (419, 675),
}


def validate_seed(seed: dict[str, torch.Tensor], bos_token_id: int = 1):
    assert "token_ids" in seed, "seed must contain a 'token_ids' key"
    assert "attention_mask" in seed, "seed must contain a 'attention_mask' key"
    assert seed["token_ids"].shape[0] == 1, "seed must have a batch size of 1"

    # ensure the seed starts with a BOS token
    assert seed["token_ids"][0, 0] == bos_token_id, "seed must start with a BOS token"

    pitches = seed["token_ids"][:, 1::4]
    velocities = seed["token_ids"][:, 2::4]
    durations = seed["token_ids"][:, 3::4]
    time_shifts = seed["token_ids"][:, 4::4]

    # assert sequences in expected ranges
    assert torch.all(pitches >= TOKEN_RANGES["pitch"][0]), "pitch out of range"
    assert torch.all(pitches <= TOKEN_RANGES["pitch"][1]), "pitch out of range"
    assert torch.all(velocities >= TOKEN_RANGES["velocity"][0]), "velocity out of range"
    assert torch.all(velocities <= TOKEN_RANGES["velocity"][1]), "velocity out of range"
    assert torch.all(durations >= TOKEN_RANGES["duration"][0]), "duration out of range"
    assert torch.all(durations <= TOKEN_RANGES["duration"][1]), "duration out of range"
    assert torch.all(time_shifts >= TOKEN_RANGES["timeshift"][0]), "time_shift out of range"
    assert torch.all(time_shifts <= TOKEN_RANGES["timeshift"][1]), "time_shift out of range"


def generate_from_seed(
    model: StructuredTransformer,
    input_seed: dict[str, torch.Tensor] = BOS_SEED,
    temperature: float = 1.0,
    top_k_fraction: float = 0.9,
    min_steps: int = 1,
    steps: int = 1,
    device: str | torch.device = "cpu",
):
    validate_seed(input_seed, bos_token_id=model.config.bos_token_id)
    logger.info("generating sequence")
    assert min_steps <= steps, "min_steps must be less than or equal to steps"

    device = torch.device(device)
    generated_sequence = {}
    for k, v in input_seed.items():
        generated_sequence[k] = v.clone().detach().to(device)

    model = model.to(device)

    token_types = ["pitch", "velocity", "duration", "time_shift"]
    token_type_masks = {
        "pitch": model.pitch_mask[None, :],
        "velocity": model.velocity_mask[None, :],
        "duration": model.duration_mask[None, :],
        "time_shift": model.time_shift_mask[None, :],
    }

    # keep an index to know which token type we're currently generating
    current_idx = generated_sequence["token_ids"].shape[1] - 1

    # input_seq shape: (batch_dim, seq_len)
    for step in range(steps):
        # figure out which token type we're currently generating
        token_type = token_types[current_idx % len(token_types)]
        current_idx += 1

        logits = model(generated_sequence)
        # logits shape: (batch_size, seq_len, vocab_size)
        next_token_logits = logits[:, -1, :]
        if step <= min_steps:
            # if we haven't generated the minimum number of steps, mask out the eos token
            next_token_logits[:, model.config.eos_token_id] = -float("inf")
        # mask out invalid tokens according to the structured sequence rules
        current_mask = token_type_masks[token_type]
        if current_mask is not None:
            next_token_logits.masked_fill_(current_mask, -float("inf"))
        # next_token_logits shape: (batch_size, vocab_size)
        next_token_logits = top_k(next_token_logits / temperature, frac_num_tokens=top_k_fraction)
        # normalize scaled logits to get predicted probabilities
        sample_distribution = torch.softmax(next_token_logits, dim=1)
        # sample from distribution to get predicted output token
        next_input = torch.multinomial(sample_distribution, num_samples=1)
        # next_input shape: (batch_size, num_samples)
        # add the output to the context window that we repeatedly feed the model
        generated_sequence["token_ids"] = torch.concat((generated_sequence["token_ids"], next_input), dim=1)

        generated_sequence["attention_mask"] = torch.cat(
            (generated_sequence["attention_mask"], torch.tensor([[True]]).to(device)),
            dim=1,
        )

        # check if we predicted the end of sequence token
        if generated_sequence["token_ids"][:, -1] == model.config.eos_token_id:
            logger.info(f"generated {step} steps before predicting end of sequence")
            break
    return generated_sequence
