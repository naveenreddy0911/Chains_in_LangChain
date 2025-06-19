# Chains in LangChain

This repository demonstrates how to implement and use **different types of chains** in LangChain. Chaining lets you connect components together — prompts, parsers, and models — to create more sophisticated workflows. 
This project covers **simple chains**, **sequential chains**, **parallel chains**, and **conditional chains**. These techniques help you build flexible and reusable components for your applications.


## 📄 File Descriptions

- `simple_chain.py`  Creates a **simple chain**.

- `sequential_chain.py`  Demonstrates **SequentialChain**.

- `parallel_chain.py`  Shows how to execute chains in **parallel**.

- `conditional_chain.py`  Demonstrates **Conditional Chain**.

## Environment Variables

To use this project, you need to set API keys in a `.env` file in the root directory.

### 📄 .env file format

```env
OPENAI_API_KEY="your_openai_key"
ANTHROPIC_API_KEY="your_anthropic_key"
GOOGLE_API_KEY="your_google_key"
HUGGINGFACEHUB_API_KEY="your_huggingfacehub_key"
```

### Dont have closed-source API key ? 
Refer to the repo [**Models_in_LangChain**](https://github.com/naveenreddy0911/Models_in_LangChain) for using open-sourced Hugging Face chat models (via API or locally).
