#!/usr/bin/env python3
"""
Simple Medical Dialog Agent
A lightweight medical conversation assistant using Meditron-7B

Usage:
    python medical_agent.py

Requirements:
    pip install transformers torch accelerate
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import sys
from datetime import datetime

class MedicalAgent:
    def __init__(self, model_name="BioMistral/BioMistral-7B", use_4bit=True):
        """
        Initialize the medical agent
        
        Args:
            model_name: HuggingFace model ID (default: Meditron-7B)
            use_4bit: Use 4-bit quantization for lower VRAM (default: True)
        """
        print("=" * 60)
        print("Medical Dialog Agent - Initializing...")
        print("=" * 60)
        print(f"Model: {model_name}")
        print(f"4-bit Quantization: {use_4bit}")
        print()
        
        # Check for GPU
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Device: {self.device}")
        
        if self.device == "cpu":
            print("⚠️  Warning: Running on CPU. This will be slow.")
            print("   For better performance, use a GPU with CUDA.\n")
        
        # Load tokenizer
        print("Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Configure model loading
        if use_4bit and self.device == "cuda":
            print("Loading model with 4-bit quantization (low VRAM mode)...")
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=quantization_config,
                device_map="auto",
                trust_remote_code=True
            )
        else:
            print("Loading model (standard mode)...")
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map="auto" if self.device == "cuda" else None,
                trust_remote_code=True
            )
            if self.device == "cpu":
                self.model = self.model.to(self.device)
        
        print("✓ Model loaded successfully!\n")
        
        # Conversation history
        self.conversation_history = []
        
    def generate_response(self, user_input, max_length=300, temperature=0.7):
        """
        Generate a response to user input
        
        Args:
            user_input: User's message
            max_length: Maximum response length
            temperature: Sampling temperature (higher = more creative)
        
        Returns:
            Generated response text
        """
        # Build prompt with conversation history
        if len(self.conversation_history) > 0:
            context = "\n".join([
                f"Patient: {msg['user']}\nDoctor: {msg['assistant']}" 
                for msg in self.conversation_history[-3:]  # Last 3 exchanges
            ])
            prompt = f"{context}\nPatient: {user_input}\nDoctor:"
        else:
            prompt = f"Patient: {user_input}\nDoctor:"
        
        # Tokenize
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_length,
                temperature=temperature,
                do_sample=True,
                top_p=0.9,
                top_k=50,
                pad_token_id=self.tokenizer.eos_token_id,
                repetition_penalty=1.2
            )
        
        # Decode response
        full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract only the assistant's response
        if "Doctor:" in full_response:
            response = full_response.split("Doctor:")[-1].strip()
            # Clean up if it contains next patient statement
            if "Patient:" in response:
                response = response.split("Patient:")[0].strip()
        else:
            response = full_response.strip()
        
        return response
    
    def chat(self, user_input):
        """
        Main chat interface
        
        Args:
            user_input: User's message
        
        Returns:
            Agent's response
        """
        if not user_input.strip():
            return "Please enter a message."
        
        # Generate response
        response = self.generate_response(user_input)
        
        # Store in history
        self.conversation_history.append({
            "user": user_input,
            "assistant": response,
            "timestamp": datetime.now().isoformat()
        })
        
        return response
    
    def clear_history(self):
        """Clear conversation history"""
        self.conversation_history = []
        print("Conversation history cleared.\n")
    
    def save_conversation(self, filename=None):
        """Save conversation to file"""
        if not filename:
            filename = f"medical_conversation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        
        with open(filename, 'w') as f:
            f.write("Medical Conversation Log\n")
            f.write("=" * 60 + "\n\n")
            for msg in self.conversation_history:
                f.write(f"[{msg['timestamp']}]\n")
                f.write(f"Patient: {msg['user']}\n")
                f.write(f"Doctor: {msg['assistant']}\n\n")
        
        print(f"✓ Conversation saved to: {filename}\n")
        return filename


def print_welcome():
    """Print welcome message"""
    print("\n" + "=" * 60)
    print("          MEDICAL DIALOG AGENT")
    print("=" * 60)
    print("\n⚠️  DISCLAIMER: This is for educational/research purposes only.")
    print("   NOT a substitute for professional medical advice.\n")
    print("Commands:")
    print("  - Type your symptoms or questions")
    print("  - 'clear' - Clear conversation history")
    print("  - 'save' - Save conversation to file")
    print("  - 'quit' or 'exit' - Exit the program\n")
    print("=" * 60 + "\n")


def main():
    """Main function to run the medical agent"""
    
    # Initialize agent
    try:
        agent = MedicalAgent(
            model_name="BioMistral/BioMistral-7B",  # Use "./biomistral-7b" for local
            use_4bit=True  # Set to False if you have plenty of VRAM
        )
    except Exception as e:
        print(f"\n✗ Error loading model: {e}")
        print("\nMake sure you have installed the requirements:")
        print("  pip install transformers torch accelerate bitsandbytes")
        print("\nAnd have enough disk space (~14GB) for model download.")
        sys.exit(1)
    
    # Print welcome message
    print_welcome()
    
    # Main conversation loop
    while True:
        try:
            # Get user input
            user_input = input("You: ").strip()
            
            if not user_input:
                continue
            
            # Handle commands
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("\nThank you for using Medical Dialog Agent!")
                if len(agent.conversation_history) > 0:
                    save = input("Save conversation before exiting? (y/n): ").lower()
                    if save == 'y':
                        agent.save_conversation()
                print("Goodbye!\n")
                break
            
            elif user_input.lower() == 'clear':
                agent.clear_history()
                continue
            
            elif user_input.lower() == 'save':
                agent.save_conversation()
                continue
            
            # Generate response
            print("\nAgent: ", end="", flush=True)
            response = agent.chat(user_input)
            print(response + "\n")
            
        except KeyboardInterrupt:
            print("\n\nInterrupted by user.")
            break
        except Exception as e:
            print(f"\n✗ Error: {e}\n")
            continue


if __name__ == "__main__":
    main()
