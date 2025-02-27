"""                                                                                                                                                                                   
Distillation module for LLM optimization.                                                                                                                                             
"""                                                                                                                                                                                   
                                                                                                                                                                                    
import os                                                                                                                                                                             
import logging                                                                                                                                                                        
from typing import Dict, Any, Optional, Tuple                                                                                                                                         
                                                                                                                                                                                    
import torch                                                                                                                                                                          
import torch.nn as nn                                                                                                                                                                 
import torch.nn.functional as F                                                                                                                                                       
from torch.utils.data import DataLoader                                                                                                                                               
from transformers import (                                                                                                                                                            
    AutoModelForCausalLM,                                                                                                                                                             
    AutoTokenizer,                                                                                                                                                                    
    Trainer,                                                                                                                                                                          
    TrainingArguments                                                                                                                                                                 
)                                                                                                                                                                                     
from datasets import load_dataset                                                                                                                                                     
                                                                                                                                                                                    
from llm_optimizer.base import OptimizationStage                                                                                                                                      
from llm_optimizer.utils.model import load_model_and_tokenizer                                                                                                                        
                                                                                                                                                                                    
logger = logging.getLogger(__name__)                                                                                                                                                  
                                                                                                                                                                                    
                                                                                                                                                                                    
class DistillationStage(OptimizationStage):                                                                                                                                           
    """                                                                                                                                                                               
    Distillation stage for LLM optimization.                                                                                                                                          
                                                                                                                                                                                    
    Implements knowledge distillation from a teacher model to a smaller student model.                                                                                                
    """                                                                                                                                                                               
                                                                                                                                                                                    
    def run(self, model_state: Dict[str, Any]) -> Dict[str, Any]:                                                                                                                     
        """                                                                                                                                                                           
        Run the distillation stage.                                                                                                                                                   
                                                                                                                                                                                    
        Args:                                                                                                                                                                         
            model_state: Current state of the model (teacher model)                                                                                                                   
                                                                                                                                                                                    
        Returns:                                                                                                                                                                      
            Updated model state and metrics                                                                                                                                           
        """                                                                                                                                                                           
        self.validate_input(model_state)                                                                                                                                              
                                                                                                                                                                                    
        # Extract configuration                                                                                                                                                       
        teacher_model_path = model_state["model_path"]                                                                                                                                
        student_model_name = self.config.get("student_model", "distilgpt2")                                                                                                           
        output_dir = os.path.join(                                                                                                                                                    
            self.config.get("output_dir", "./outputs"),                                                                                                                               
            f"distilled_{os.path.basename(student_model_name)}"                                                                                                                       
        )                                                                                                                                                                             
                                                                                                                                                                                    
        logger.info(f"Distilling from {teacher_model_path} to {student_model_name}")                                                                                                  
                                                                                                                                                                                    
        # Load teacher model and tokenizer                                                                                                                                            
        teacher_model, teacher_tokenizer = load_model_and_tokenizer(teacher_model_path)                                                                                               
                                                                                                                                                                                    
        # Load student model and tokenizer                                                                                                                                            
        student_model, student_tokenizer = load_model_and_tokenizer(student_model_name)                                                                                               
                                                                                                                                                                                    
        # Ensure student uses the same tokenizer as teacher for compatibility                                                                                                         
        student_tokenizer = teacher_tokenizer                                                                                                                                         
                                                                                                                                                                                    
        # Load dataset                                                                                                                                                                
        dataset_name = self.config.get("dataset", "wikitext")                                                                                                                         
        dataset_config = self.config.get("dataset_config", "wikitext-2-raw-v1")                                                                                                       
                                                                                                                                                                                    
        logger.info(f"Loading dataset {dataset_name}/{dataset_config}")                                                                                                               
        dataset = load_dataset(dataset_name, dataset_config)                                                                                                                          
                                                                                                                                                                                    
        # Prepare dataset                                                                                                                                                             
        tokenized_dataset = self._prepare_dataset(dataset, teacher_tokenizer)                                                                                                         
                                                                                                                                                                                    
        # Perform distillation                                                                                                                                                        
        distilled_model = self._distill_model(                                                                                                                                        
            teacher_model,                                                                                                                                                            
            student_model,                                                                                                                                                            
            tokenized_dataset                                                                                                                                                         
        )                                                                                                                                                                             
                                                                                                                                                                                    
        # Save the distilled model                                                                                                                                                    
        os.makedirs(output_dir, exist_ok=True)                                                                                                                                        
        distilled_model.save_pretrained(output_dir)                                                                                                                                   
        student_tokenizer.save_pretrained(output_dir)                                                                                                                                 
                                                                                                                                                                                    
        # Save distillation configuration                                                                                                                                             
        with open(os.path.join(output_dir, "distillation_config.txt"), "w") as f:                                                                                                     
            f.write(f"Teacher model: {teacher_model_path}\n")                                                                                                                         
            f.write(f"Student model: {student_model_name}\n")                                                                                                                         
            f.write(f"Dataset: {dataset_name}/{dataset_config}\n")                                                                                                                    
                                                                                                                                                                                    
            # Add model size comparison                                                                                                                                               
            teacher_size = self._get_model_size(teacher_model)                                                                                                                        
            student_size = self._get_model_size(distilled_model)                                                                                                                      
            size_reduction = (teacher_size - student_size) / teacher_size * 100                                                                                                       
                                                                                                                                                                                    
            f.write(f"Teacher size: {teacher_size:.2f} MB\n")                                                                                                                         
            f.write(f"Student size: {student_size:.2f} MB\n")                                                                                                                         
            f.write(f"Size reduction: {size_reduction:.2f}%\n")                                                                                                                       
                                                                                                                                                                                    
        logger.info(f"Distilled model saved to {output_dir}")                                                                                                                         
                                                                                                                                                                                    
        # Return updated model state and metrics                                                                                                                                      
        return {                                                                                                                                                                      
            "model_state": {                                                                                                                                                          
                "model_path": output_dir,                                                                                                                                             
                "is_pretrained": True,                                                                                                                                                
                "distilled_from": teacher_model_path,                                                                                                                                 
            },                                                                                                                                                                        
            "metrics": {                                                                                                                                                              
                "teacher_size_mb": teacher_size,                                                                                                                                      
                "student_size_mb": student_size,                                                                                                                                      
                "size_reduction_percent": size_reduction,                                                                                                                             
                "perplexity": self._evaluate_perplexity(distilled_model, student_tokenizer, tokenized_dataset["validation"]),                                                         
            },                                                                                                                                                                        
            "artifacts": {                                                                                                                                                            
                "model_path": output_dir,                                                                                                                                             
                "tokenizer_path": output_dir,                                                                                                                                         
                "config_path": os.path.join(output_dir, "distillation_config.txt"),                                                                                                   
            }                                                                                                                                                                         
        }                                                                                                                                                                             
                                                                                                                                                                                    
    def _prepare_dataset(self, dataset, tokenizer):                                                                                                                                   
        """Prepare and tokenize the dataset."""                                                                                                                                       
        max_length = self.config.get("max_length", 512)                                                                                                                               
                                                                                                                                                                                    
        def tokenize_function(examples):                                                                                                                                              
            return tokenizer(                                                                                                                                                         
                examples["text"],                                                                                                                                                     
                truncation=True,                                                                                                                                                      
                max_length=max_length,                                                                                                                                                
                padding="max_length",                                                                                                                                                 
            )                                                                                                                                                                         
                                                                                                                                                                                    
        tokenized_dataset = dataset.map(                                                                                                                                              
            tokenize_function,                                                                                                                                                        
            batched=True,                                                                                                                                                             
            num_proc=4,                                                                                                                                                               
            remove_columns=["text"],                                                                                                                                                  
        )                                                                                                                                                                             
                                                                                                                                                                                    
        return tokenized_dataset                                                                                                                                                      
                                                                                                                                                                                    
    def _distill_model(self, teacher_model, student_model, tokenized_dataset):                                                                                                        
        """                                                                                                                                                                           
        Perform knowledge distillation from teacher to student.                                                                                                                       
                                                                                                                                                                                    
        Args:                                                                                                                                                                         
            teacher_model: The teacher model                                                                                                                                          
            student_model: The student model                                                                                                                                          
            tokenized_dataset: The dataset for distillation                                                                                                                           
                                                                                                                                                                                    
        Returns:                                                                                                                                                                      
            Distilled student model                                                                                                                                                   
        """                                                                                                                                                                           
        # Define distillation parameters                                                                                                                                              
        temperature = self.config.get("temperature", 2.0)                                                                                                                             
        alpha = self.config.get("alpha", 0.5)  # Weight for distillation loss                                                                                                         
                                                                                                                                                                                    
        # Create a custom trainer for distillation                                                                                                                                    
        class DistillationTrainer(Trainer):                                                                                                                                           
            def __init__(self, *args, teacher_model=None, alpha=0.5, temperature=2.0, **kwargs):                                                                                      
                super().__init__(*args, **kwargs)                                                                                                                                     
                self.teacher_model = teacher_model                                                                                                                                    
                self.alpha = alpha                                                                                                                                                    
                self.temperature = temperature                                                                                                                                        
                                                                                                                                                                                    
                # Move teacher to same device as student                                                                                                                              
                self.teacher_model.to(self.model.device)                                                                                                                              
                self.teacher_model.eval()                                                                                                                                             
                                                                                                                                                                                    
            def compute_loss(self, model, inputs, return_outputs=False):                                                                                                              
                # Compute student outputs and loss                                                                                                                                    
                outputs = model(**inputs)                                                                                                                                             
                student_loss = outputs.loss                                                                                                                                           
                                                                                                                                                                                    
                # Get student logits                                                                                                                                                  
                student_logits = outputs.logits                                                                                                                                       
                                                                                                                                                                                    
                # Compute teacher logits                                                                                                                                              
                with torch.no_grad():                                                                                                                                                 
                    teacher_outputs = self.teacher_model(**inputs)                                                                                                                    
                    teacher_logits = teacher_outputs.logits                                                                                                                           
                                                                                                                                                                                    
                # Compute distillation loss (KL divergence)                                                                                                                           
                distillation_loss = self._distillation_loss(                                                                                                                          
                    student_logits,                                                                                                                                                   
                    teacher_logits,                                                                                                                                                   
                    self.temperature                                                                                                                                                  
                )                                                                                                                                                                     
                                                                                                                                                                                    
                # Combine losses                                                                                                                                                      
                loss = (1 - self.alpha) * student_loss + self.alpha * distillation_loss                                                                                               
                                                                                                                                                                                    
                return (loss, outputs) if return_outputs else loss                                                                                                                    
                                                                                                                                                                                    
            def _distillation_loss(self, student_logits, teacher_logits, temperature):                                                                                                
                """Compute the knowledge distillation loss."""                                                                                                                        
                # Soften probabilities and compute KL divergence                                                                                                                      
                soft_targets = F.softmax(teacher_logits / temperature, dim=-1)                                                                                                        
                soft_prob = F.log_softmax(student_logits / temperature, dim=-1)                                                                                                       
                                                                                                                                                                                    
                # KL divergence loss                                                                                                                                                  
                kl_div = F.kl_div(                                                                                                                                                    
                    soft_prob,                                                                                                                                                        
                    soft_targets,                                                                                                                                                     
                    reduction="batchmean"                                                                                                                                             
                ) * (temperature ** 2)                                                                                                                                                
                                                                                                                                                                                    
                return kl_div                                                                                                                                                         
                                                                                                                                                                                    
        # Set up training arguments                                                                                                                                                   
        training_args = TrainingArguments(                                                                                                                                            
            output_dir=self.config.get("output_dir", "./outputs/distillation"),                                                                                                       
            num_train_epochs=self.config.get("num_epochs", 3),                                                                                                                        
            per_device_train_batch_size=self.config.get("batch_size", 4),                                                                                                             
            gradient_accumulation_steps=self.config.get("gradient_accumulation_steps", 4),                                                                                            
            learning_rate=self.config.get("learning_rate", 5e-5),                                                                                                                     
            weight_decay=self.config.get("weight_decay", 0.01),                                                                                                                       
            warmup_steps=self.config.get("warmup_steps", 500),                                                                                                                        
            logging_dir="./logs",                                                                                                                                                     
            logging_steps=self.config.get("logging_steps", 100),                                                                                                                      
            save_steps=self.config.get("save_steps", 1000),                                                                                                                           
            evaluation_strategy="steps",                                                                                                                                              
            eval_steps=self.config.get("eval_steps", 500),                                                                                                                            
            save_total_limit=self.config.get("save_total_limit", 3),                                                                                                                  
            load_best_model_at_end=True,                                                                                                                                              
        )                                                                                                                                                                             
                                                                                                                                                                                    
        # Initialize and run the distillation trainer                                                                                                                                 
        trainer = DistillationTrainer(                                                                                                                                                
            model=student_model,                                                                                                                                                      
            args=training_args,                                                                                                                                                       
            train_dataset=tokenized_dataset["train"],                                                                                                                                 
            eval_dataset=tokenized_dataset["validation"],                                                                                                                             
            teacher_model=teacher_model,                                                                                                                                              
            alpha=alpha,                                                                                                                                                              
            temperature=temperature,                                                                                                                                                  
        )                                                                                                                                                                             
                                                                                                                                                                                    
        trainer.train()                                                                                                                                                               
                                                                                                                                                                                    
        return student_model                                                                                                                                                          
                                                                                                                                                                                    
    def _evaluate_perplexity(self, model, tokenizer, eval_dataset):                                                                                                                   
        """Evaluate model perplexity on the validation dataset."""                                                                                                                    
        model.eval()                                                                                                                                                                  
                                                                                                                                                                                    
        eval_dataloader = torch.utils.data.DataLoader(                                                                                                                                
            eval_dataset, batch_size=self.config.get("eval_batch_size", 4)                                                                                                            
        )                                                                                                                                                                             
                                                                                                                                                                                    
        total_loss = 0.0                                                                                                                                                              
        total_tokens = 0                                                                                                                                                              
                                                                                                                                                                                    
        with torch.no_grad():                                                                                                                                                         
            for batch in eval_dataloader:                                                                                                                                             
                input_ids = batch["input_ids"].to(model.device)                                                                                                                       
                attention_mask = batch["attention_mask"].to(model.device)                                                                                                             
                                                                                                                                                                                    
                outputs = model(                                                                                                                                                      
                    input_ids=input_ids,                                                                                                                                              
                    attention_mask=attention_mask,                                                                                                                                    
                    labels=input_ids,                                                                                                                                                 
                )                                                                                                                                                                     
                                                                                                                                                                                    
                loss = outputs.loss                                                                                                                                                   
                total_loss += loss.item() * input_ids.size(0)                                                                                                                         
                total_tokens += input_ids.size(0)                                                                                                                                     
                                                                                                                                                                                    
        avg_loss = total_loss / total_tokens                                                                                                                                          
        perplexity = torch.exp(torch.tensor(avg_loss)).item()                                                                                                                         
                                                                                                                                                                                    
        return perplexity                                                                                                                                                             
                                                                                                                                                                                    
    def _get_model_size(self, model):                                                                                                                                                 
        """Calculate the size of a model in MB."""                                                                                                                                    
        param_size = 0                                                                                                                                                                
        for param in model.parameters():                                                                                                                                              
            param_size += param.nelement() * param.element_size()                                                                                                                     
                                                                                                                                                                                    
        buffer_size = 0                                                                                                                                                               
        for buffer in model.buffers():                                                                                                                                                
            buffer_size += buffer.nelement() * buffer.element_size()                                                                                                                  
                                                                                                                                                                                    
        size_mb = (param_size + buffer_size) / 1024 / 1024                                                                                                                            
                                                                                                                                                                                    
        return size_mb  