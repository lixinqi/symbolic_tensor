import torch
def get_causal_attention_mask(
    token_mask: torch.Tensor,
) -> torch.Tensor:
    """Generate a causal attention mask from a token mask.
    Args:
        token_mask: Bool tensor of shape (batch_size, seq_len).
    Returns:
        Bool tensor of shape (batch_size, seq_len, seq_len) where
        mask[b, i, j] is True iff j <= i and token_mask[b, i] is True.
    """
    batch_size, seq_len = token_mask.shape
    tril = torch.tril(
        torch.ones(seq_len, seq_len, dtype=torch.bool, device=token_mask.device)
    )
    expanded_token_mask = token_mask[:, :, None].expand(-1, -1, seq_len)
    return tril & expanded_token_mask
if __name__ == "__main__":
    print("Running get_causal_attention_mask tests...\n")
    def run_test(name: str, condition: bool, expected=None, actual=None):
        if condition:
            print(f"  \u2713 {name}")
        else:
            print(f"  \u2717 {name}")
            if expected is not None and actual is not None:
                print(f"    expected: {expected}")
                print(f"    actual:   {actual}")
    print("Test 1: All valid tokens, causal mask")
    mask = torch.ones(1, 3, dtype=torch.bool)
    result = get_causal_attention_mask(mask)
    expected = torch.tensor([[[True, False, False],
                              [True, True,  False],
                              [True, True,  True]]])
    run_test("shape is (1, 3, 3)", result.shape == (1, 3, 3))
    run_test("matches lower triangular", torch.equal(result, expected),
             expected=expected, actual=result)
    print("Test 2: Last token padded")
    mask = torch.tensor([[True, True, False]])
    result = get_causal_attention_mask(mask)
    expected = torch.tensor([[[True, False, False],
                              [True, True,  False],
                              [False, False, False]]])
    run_test("shape is (1, 3, 3)", result.shape == (1, 3, 3))
    run_test("padding row is all False", not result[0, 2].any())
    run_test("matches expected", torch.equal(result, expected),
             expected=expected, actual=result)
    print("Test 3: Batch of 2")
    mask = torch.tensor([[True, True, True],
                         [True, False, False]])
    result = get_causal_attention_mask(mask)
    run_test("shape is (2, 3, 3)", result.shape == (2, 3, 3))
    run_test("batch 0 is full causal",
             torch.equal(result[0], torch.tril(torch.ones(3, 3, dtype=torch.bool))))
    run_test("batch 1 row 0 is [T,F,F]",
             torch.equal(result[1, 0], torch.tensor([True, False, False])))
    run_test("batch 1 rows 1,2 all False",
             not result[1, 1:].any())
    print("Test 4: Single token")
    mask = torch.ones(1, 1, dtype=torch.bool)
    result = get_causal_attention_mask(mask)
    run_test("shape is (1, 1, 1)", result.shape == (1, 1, 1))
    run_test("value is True", result.item() is True)
    print("Test 5: All tokens masked")
    mask = torch.zeros(1, 3, dtype=torch.bool)
    result = get_causal_attention_mask(mask)
    run_test("all False", not result.any())
    print("Test 6: Output dtype")
    mask = torch.ones(2, 4, dtype=torch.bool)
    result = get_causal_attention_mask(mask)
    run_test("dtype is bool", result.dtype == torch.bool)
    run_test("shape is (2, 4, 4)", result.shape == (2, 4, 4))
    print("\nAll tests completed.")