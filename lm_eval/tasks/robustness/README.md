# ROBUSTNESS

## License

Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

## References
[1] Wang, Yubo, et al. "Mmlu-pro: A more robust and challenging multi-task language understanding benchmark." arXiv preprint arXiv:2406.01574 (2024).

[2] Zhong, Wanjun, et al. "Agieval: A human-centric benchmark for evaluating foundation models." arXiv preprint arXiv:2304.06364 (2023).


## Citation
```bib
[Citation placeholder]
```

## Groups

- `robustness_mmlu_pro`: three 0-shot robutstness tasks on MMLU-PRO dataset

- `robustness_agieval`: three 0-shot robutstness tasks on the AGIEVAL datasets multiple choice questions subsets:  `'agieval-sat-math'`, `'agieval-lsat-lr'`, `'agieval-lsat-rc'`, `'agieval-logiqa-en'`, `'agieval-aqua-rat'`, `'agieval-sat-en'`, `'agieval-lsat-ar'` 

- `robustness_mmlu_pro_fewshot`: a 5-shot robutstness task on MMLU-PRO dataset

## Tasks

Both `robustness_mmlu_pro` and `robustness_agieval` contain the following 3 tasks:

* Option format robustness: `option_format_robustness_mmlu_pro`, `option_format_robustness_agieval`

* Option order robustness: 
`option_order_robustness_mmlu_pro`, `option_order_robustness_agieval`

* Prompt robustness: 
`prompt_robustness_mmlu_pro`, 
`prompt_robustness_agieval`

### Option format robustness

Measures the model's robustness to the following option formats:
<br>
- Uppercase latter + colon (`A: option_1` `B: option_2` ... )
- Uppercase latter + parenthesis (`A) option_1` `B) option_2` ... )
- Uppercase latter + dot (`A. option_1` `B. option_2` ... )
- Numeral + dot (`1. option_1` `2. option_2` ... )
- Lowercase latter + colon (`a: option_1` `b: option_2` ... )
- Roman numeral + colon (`I: option_1` `II: option_2` ... )

### Option order robustness

Measures the model's robustness to the placment of the correct answer in the options list by swapping the correct answer with all the other possible options.

### Prompt robustness

Measures the model's robustness to 10 different prompts. list of the prompts can be found in the `./prompt_templates.json` file under the key `prompt_robustness`.

### Fewshot Prompt robustness

Measures the model's robustness towards the same 10 prompts as the former task in a fewshot manner. This task is only available for MMLU-PRO.

For evaluating `robustness_mmlu_pro_fewshot` it is required to pass the `"--fewshot_as_multiturn"` flag as well as set the `"--num_fewshot 5"`,

## Notes

- All The tasks are designed for **Instruct** models for which we recommend to pass "`--apply_chat_template`" flag.


## Checklist

For adding novel benchmarks/datasets to the library:
* [-] Is the task an existing benchmark in the literature?
  * [-] Have you referenced the original paper that introduced the task? - Will be referenced as soon as the paper is published
  * [ ] If yes, does the original paper provide a reference implementation? If so, have you checked against the reference implementation and documented how to run such a test?


If other tasks on this dataset are already supported:
* [x] Is the "Main" variant of this task clearly denoted?
* [x] Have you provided a short sentence in a README on what each new variant adds / evaluates?
* [x] Have you noted which, if any, published evaluation setups are matched by this variant?