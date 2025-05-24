# Scientific Title Generation using transformer-based seq2seq models

## About the project
This project ultilitzes transformer-based seq2seq models from [Hugging Face](https://huggingface.co/) to generate titles for a given abstract of a scientific publication, preferably in the field of machine learning. The project was assigned as a part of the course **Text Mining** at the University of Science - VNUHCM.

---

## Team members

| Name | Student ID |
| --- | --- |
| Khuu Thanh Thien | 22127396 |
| Huynh Thien Thuan | 22127407 |
| Huynh Nhat Nam | 22127282 |

---

### 1. Data Source
The data source is obtained from **Springer Nature** using `requests`and `BeautifulSoup`.

Access all links to the data [here](./springer_journal_data_url.txt)

### 2. Model

- `google/flan-t5-base`: created for performing various tasks, good for instruction fine-tuning.
- `facebook/bart-base`: made for semantic comprehension, optimized for text generation.

Links to the complete models:
- [tiam4tt/flan-t5-titlegen-springer](https://huggingface.co/tiam4tt/flan-t5-titlegen-springer)
- [HTThuanHcmus/bart-finetune-scientific-improve](https://huggingface.co/HTThuanHcmus/bart-finetune-scientific-improve)

### 4. Model training

Training is carried out on **Kaggle** with the use of **Tesla P100-PCIE-16GB** GPU.

Methodology:
- **Instruction fine-tune:** concatenate the abstract and a prompt to form a single input.
- **Keyword aware instruction fine-tune:** concatenate the abstract, a prompt, and keywords to form a single input.


### 4. Deployment

Hosted on [Hugging Face Spaces](https://huggingface.co/spaces/tiam4tt/title-generator-for-Machine-Learning-publications).

### 5. Running locally

Navigate to project directory
```bash
cd src/
```

Install required packages
```bash
pip install -r requirements.txt
```

Run the app
```bash
streamlit run app.py
```

### 6. Using the models in your own code
Here's and example:
```python
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

model = AutoModelForSeq2SeqLM.from_pretrained("tiam4tt/flan-t5-titlegen-springer")
tokenizer = AutoTokenizer.from_pretrained("tiam4tt/flan-t5-titlegen-springer")

abstract = "Transfer learning has become a crucial technique in deep learning, enabling models to leverage knowledge from pre-trained networks for improved performance on new tasks. In this study, we propose an optimized fine-tuning strategy for convolutional neural networks (CNNs), reducing training time while maintaining high accuracy. Experiments on CIFAR-10 show a 15% improvement in efficiency compared to standard fine-tuning methods, demonstrating the effectiveness of our approach."

inputs = tokenizer(abstract, return_tensors="pt", padding=True, truncation=True)
outputs = model.generate(**inputs, max_new_tokens=32)

title = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(title)
```