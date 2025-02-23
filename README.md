# WOOGPT  

**WOOGPT** is a custom transformer-based language model trained on *The Wizard of Oz*, designed to generate creative, domain-specific responses that capture the magic and whimsy of Oz.  

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)  [![Python Version](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/downloads/)  

## Overview  

WOOGPT is a PyTorch-based character-level GPT model leveraging **multi-head self-attention**, **feedforward layers**, and **residual connections**. Fine-tuned exclusively on *The Wizard of Oz*, it provides engaging and context-aware responses within its domain.  

## Features  

- 🎭 **Thematic Q&A:** Expertly answers questions about *The Wizard of Oz*.  
- 💬 **Interactive Chat Agent:** Offers an intuitive and engaging conversation experience.  
- ⚡ **Custom Transformer Architecture:** Built from scratch with configurable hyperparameters.  
 

## Inference Agent  

Run the inference agent to interact with WOOGPT:  

```bash
python agent1.py
```
Enter your queries about **The Wizard of Oz**, and WOOGPT will respond in character.

## Demonstration

Here’s how **WOOGPT** responds based on the type of question asked:

### 🎭 Wizard of Oz-Related Questions  
When you ask about *The Wizard of Oz*, WOOGPT provides fun, engaging, and domain-specific answers:  
![WOOGPT Demo](src/assets/clip1.gif)  

### ❌ Unrelated Questions  
If you ask something outside the world of Oz, WOOGPT stays in character and gently redirects you back to the magical land:  
![WOOGPT](src/assets/clip2.gif) 


## Acknowledgements
* L. Frank Baum – The Wizard of Oz
* Hugging Face Transformers – https://huggingface.co/docs/transformers/main/en/index
* Background - Ido Hadary