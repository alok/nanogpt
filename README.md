# NanoGPT

This is an implementation of the decoder-only GPT architecture, specifically GPT 2. It's based on [Andrej Karpathy's NanoGPT](https://www.youtube.com/watch?v=kCc8FmEb1nY), trained on the Tiny Shakespeare dataset.

## Differences

- No dropout is added. Although it's technically added, the drop fraction is a trivial 0%.

- [Einops](https://github.com/arogozhnikov/einops) is heavily used to avoid reshaping and transposing. It also simplifies the addition of multi-head attention.

- Fixed a couple of bugs from Andrej Karpathy's video.

- [Jaxtyping](https://github.com/google/jaxtyping) is used in places to provide shape information, though there are instances where it interacts poorly with Einops. In such cases, comments are used instead.

- We leverage the excellent [Tyro library](https://github.com/brentyi/tyro) for argument parsing. Please note that it won't work running in VSCode's Python notebook interface since VSCode hijacks argparsing. Argparse doesn't work either in this case.
