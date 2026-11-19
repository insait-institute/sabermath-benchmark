[Docs](README.md) › **Command line**

# Command line

The `sabermath` command runs the benchmark without writing any Python. It comes with the package:

```bash
pip install sabermath                 # or "sabermath[vllm]" for --vllm, see Installation
```

`python -m sabermath` does the same as `sabermath`.

```bash
sabermath evaluate BAAI/bge-m3
```

The command line covers the common case: a Hugging Face model id, loaded with Sentence Transformers or vLLM.
For everything else, such as custom processors, instructions, prompts or saving the report, use
[`evaluate()`](evaluate.md) from Python.

## `sabermath evaluate`

```text
sabermath evaluate MODEL [--task TASK]... [--device DEVICE] [--vllm]
```

| Option | Meaning | Python equivalent |
|---|---|---|
| `MODEL` | a Hugging Face model id | `evaluate(MODEL)` |
| `--task TASK` | `statement-statement`, `statement-full` or `full-full`. Repeat it to run several. Default: all three | `tasks=[...]` |
| `--device DEVICE` | device for Sentence Transformers, e.g. `cuda` or `cpu` | `init_kwargs={"device": DEVICE}` |
| `--vllm` | load the model with vLLM (needs the [`vllm` extra](installation.md#optional-vllm)) | `use_vllm=True` |

`--device` can't be combined with `--vllm`. The run uses the paper's defaults: nDCG@10 with exponential gain.

```bash
sabermath evaluate Qwen/Qwen3-Embedding-0.6B --task statement-full --device cuda
sabermath evaluate Qwen/Qwen3-Embedding-0.6B --task statement-full --task full-full --vllm
```

After the status lines and progress bar, it prints the overall nDCG@10 of each task followed by its five branches:

```text
BAAI/bge-m3 (sentence-transformers), nDCG@10

statement-full          0.5110
  Algebra               …
  Geometry              …
  Number Theory         …
  Combinatorics         …
  Calculus and Analysis …
```

## `sabermath download`

Downloads both datasets into the 🤗 Datasets cache and exits. Use it to fetch the data ahead of time, for example
on a login node before running on compute nodes that have no internet access.

```bash
sabermath download
```

## `sabermath help`

```bash
sabermath help              # list the commands
sabermath help evaluate     # options of one command
```

`-h` / `--help` works on every command too.

---

<table width="100%"><tr>
<td align="left">← <a href="api.md">API reference</a></td>
<td align="right"><a href="README.md">Docs home</a> ↩</td>
</tr></table>
