"""                                                                                                                                                                                   
Fine-tuning module for LLM optimization.                                                                                                                                              
"""                                                                                                                                                                                   
                                                                                                                                                                                    
import os                                                                                                                                                                             
import logging                                                                                                                                                                        
from typing import Dict, Any, Optional, Union                                                                                                                                         
                                                                                                                                                                                    
import torch                                                                                                                                                                          
from transformers import (                                                                                                                                                            
    AutoModelForCausalLM,                                                                                                                                                             
    AutoTokenizer,                                                                                                                                                                    
    Trainer,                                                                                                                                                                          
    TrainingArguments,                                                                                                                                                                
    DataCollatorForLanguageModeling                                                                                                                                                   
)                                                                                                                                                                                     
from datasets import load_dataset                                                                                                                                                     
from peft import (                                                                                                                                                                    
    get_peft_model,                                                                                                                                                                   
    LoraConfig,                                                                                                                                                                       
    TaskType,                                                                                                                                                                         
    PeftModel,                                                                                                                                                                        
    prepare_model_for_kbit_training                                                                                                                                                   
)                                                                                                                                                                                     
                                                                                                                                                                                    
from llm_optimizer.base import OptimizationStage                                                                                                                                      
from llm_optimizer.utils.model import load_model_and_tokenizer                                                                                                                        
                                                                                                                                                                                    
logger = logging.getLogger(__name__)                                                                                                                                                  
                                                                                                                                                                                    
                                                                                                                                                                                    
class FineTuningStage(OptimizationStage):                                                                                                                                             
    """                                                                                                                                                                               
    Fine-tuning stage for LLM optimization.                                                                                                                                           
                                                                                                                                                                                    
    Supports various fine-tuning methods:                                                                                                                                             
    - Full fine-tuning                                                                                                                                                                
    - LoRA (Low-Rank Adaptation)                                                                                                                                                      
    - QLoRA (Quantized LoRA)                                                                                                                                                          
    - P-Tuning                                                                                                                                                                        
    """                                                                                                                                                                               
                                                                                                                                                                                    
    def run(self, model_state: Dict[str, Any]) -> Dict[str, Any]:                                                                                                                     
        """                                                                                                                                                                           
        Run the fine-tuning stage.                                                                                                                                                    
                                                                                                                                                                                    
        Args:                                                                                                                                                                         
            model_state: Current state of the model                                                                                                                                   
                                                                                                                                                                                    
        Returns:                                                                                                                                                                      
            Updated model state and metrics                                                                                                                                           
        """                                                                                                                                                                           
        self.validate_input(model_state)                                                                                                                                              
                                                                                                                                                                                    
        # Extract configuration                                                                                                                                                       
        model_path = model_state["model_path"]                                                                                                                                        
        method = self.config.get("method", "lora")                                                                                                                                    
        output_dir = os.path.join(                                                                                                                                                    
            self.config.get("output_dir", "./outputs"),                                                                                                                               
            f"fine_tuned_{method}"                                                                                                                                                    
        )                                                                                                                                                                             
                                                                                                                                                                                    
        logger.info(f"Fine-tuning model {model_path} using {method}")                                                                                                                 
                                                                                                                                                                                    
        # Load dataset                                                                                                                                                                
        dataset_name = self.config.get("dataset", "wikitext")                                                                                                                         
        dataset_config = self.config.get("dataset_config", "wikitext-2-raw-v1")                                                                                                       
                                                                                                                                                                                    
        logger.info(f"Loading dataset {dataset_name}/{dataset_config}")                                                                                                               
        dataset = load_dataset(dataset_name, dataset_config)                                                                                                                          
                                                                                                                                                                                    
        # Load model and tokenizer                                                                                                                                                    
        load_in_8bit = method == "qlora"                                                                                                                                              
        load_in_4bit = method == "qlora" and self.config.get("use_4bit", False)                                                                                                       
                                                                                                                                                                                    
        model, tokenizer = load_model_and_tokenizer(                                                                                                                                  
            model_path,                                                                                                                                                               
            load_in_8bit=load_in_8bit,                                                                                                                                                
            load_in_4bit=load_in_4bit,                                                                                                                                                
            device_map="auto"                                                                                                                                                         
        )                                                                                                                                                                             
                                                                                                                                                                                    
        # Prepare dataset                                                                                                                                                             
        tokenized_dataset = self._prepare_dataset(dataset, tokenizer)                                                                                                                 
                                                                                                                                                                                    
        # Apply fine-tuning method                                                                                                                                                    
        if method == "full":                                                                                                                                                          
            fine_tuned_model = self._full_fine_tuning(model, tokenizer, tokenized_dataset)                                                                                            
        elif method == "lora":                                                                                                                                                        
            fine_tuned_model = self._lora_fine_tuning(model, tokenizer, tokenized_dataset)                                                                                            
        elif method == "qlora":                                                                                                                                                       
            fine_tuned_model = self._qlora_fine_tuning(model, tokenizer, tokenized_dataset)                                                                                           
        elif method == "ptuning":                                                                                                                                                     
            fine_tuned_model = self._ptuning_fine_tuning(model, tokenizer, tokenized_dataset)                                                                                         
        else:                                                                                                                                                                         
            raise ValueError(f"Unsupported fine-tuning method: {method}")                                                                                                             
                                                                                                                                                                                    
        # Save the fine-tuned model                                                                                                                                                   
        os.makedirs(output_dir, exist_ok=True)                                                                                                                                        
        fine_tuned_model.save_pretrained(output_dir)                                                                                                                                  
        tokenizer.save_pretrained(output_dir)                                                                                                                                         
                                                                                                                                                                                    
        logger.info(f"Fine-tuned model saved to {output_dir}")                                                                                                                        
                                                                                                                                                                                    
        # Return updated model state and metrics                                                                                                                                      
        return {                                                                                                                                                                      
            "model_state": {                                                                                                                                                          
                "model_path": output_dir,                                                                                                                                             
                "is_pretrained": True,                                                                                                                                                
                "fine_tuning_method": method,                                                                                                                                         
            },                                                                                                                                                                        
            "metrics": {                                                                                                                                                              
                "perplexity": self._evaluate_perplexity(fine_tuned_model, tokenizer, tokenized_dataset["validation"]),                                                                
            },                                                                                                                                                                        
            "artifacts": {                                                                                                                                                            
                "model_path": output_dir,                                                                                                                                             
                "tokenizer_path": output_dir,                                                                                                                                         
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
                                                                                                                                                                                    
    def _full_fine_tuning(self, model, tokenizer, tokenized_dataset):                                                                                                                 
        """Perform full fine-tuning."""                                                                                                                                               
        training_args = TrainingArguments(                                                                                                                                            
            output_dir=self.config.get("output_dir", "./outputs/full_fine_tuning"),                                                                                                   
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
                                                                                                                                                                                    
        data_collator = DataCollatorForLanguageModeling(                                                                                                                              
            tokenizer=tokenizer,                                                                                                                                                      
            mlm=False,                                                                                                                                                                
        )                                                                                                                                                                             
                                                                                                                                                                                    
        trainer = Trainer(                                                                                                                                                            
            model=model,                                                                                                                                                              
            args=training_args,                                                                                                                                                       
            train_dataset=tokenized_dataset["train"],                                                                                                                                 
            eval_dataset=tokenized_dataset["validation"],                                                                                                                             
            data_collator=data_collator,                                                                                                                                              
        )                                                                                                                                                                             
                                                                                                                                                                                    
        trainer.train()                                                                                                                                                               
                                                                                                                                                                                    
        return model                                                                                                                                                                  
                                                                                                                                                                                    
    def _lora_fine_tuning(self, model, tokenizer, tokenized_dataset):                                                                                                                 
        """Perform LoRA fine-tuning."""                                                                                                                                               
        lora_config = LoraConfig(                                                                                                                                                     
            r=self.config.get("lora_r", 16),                                                                                                                                          
            lora_alpha=self.config.get("lora_alpha", 32),                                                                                                                             
            target_modules=self.config.get("lora_target_modules", ["q_proj", "v_proj"]),                                                                                              
            lora_dropout=self.config.get("lora_dropout", 0.05),                                                                                                                       
            bias="none",                                                                                                                                                              
            task_type=TaskType.CAUSAL_LM,                                                                                                                                             
        )                                                                                                                                                                             
                                                                                                                                                                                    
        model = get_peft_model(model, lora_config)                                                                                                                                    
        model.print_trainable_parameters()                                                                                                                                            
                                                                                                                                                                                    
        training_args = TrainingArguments(                                                                                                                                            
            output_dir=self.config.get("output_dir", "./outputs/lora_fine_tuning"),                                                                                                   
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
                                                                                                                                                                                    
        data_collator = DataCollatorForLanguageModeling(                                                                                                                              
            tokenizer=tokenizer,                                                                                                                                                      
            mlm=False,                                                                                                                                                                
        )                                                                                                                                                                             
                                                                                                                                                                                    
        trainer = Trainer(                                                                                                                                                            
            model=model,                                                                                                                                                              
            args=training_args,                                                                                                                                                       
            train_dataset=tokenized_dataset["train"],                                                                                                                                 
            eval_dataset=tokenized_dataset["validation"],                                                                                                                             
            data_collator=data_collator,                                                                                                                                              
        )                                                                                                                                                                             
                                                                                                                                                                                    
        trainer.train()                                                                                                                                                               
                                                                                                                                                                                    
        return model                                                                                                                                                                  
                                                                                                                                                                                    
    def _qlora_fine_tuning(self, model, tokenizer, tokenized_dataset):                                                                                                                
        """Perform QLoRA fine-tuning."""                                                                                                                                              
        # Prepare model for k-bit training                                                                                                                                            
        model = prepare_model_for_kbit_training(model)                                                                                                                                
                                                                                                                                                                                    
        lora_config = LoraConfig(                                                                                                                                                     
            r=self.config.get("lora_r", 16),                                                                                                                                          
            lora_alpha=self.config.get("lora_alpha", 32),                                                                                                                             
            target_modules=self.config.get("lora_target_modules", ["q_proj", "v_proj"]),                                                                                              
            lora_dropout=self.config.get("lora_dropout", 0.05),                                                                                                                       
            bias="none",                                                                                                                                                              
            task_type=TaskType.CAUSAL_LM,                                                                                                                                             
        )                                                                                                                                                                             
                                                                                                                                                                                    
        model = get_peft_model(model, lora_config)                                                                                                                                    
        model.print_trainable_parameters()                                                                                                                                            
                                                                                                                                                                                    
        training_args = TrainingArguments(                                                                                                                                            
            output_dir=self.config.get("output_dir", "./outputs/qlora_fine_tuning"),                                                                                                  
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
                                                                                                                                                                                    
        data_collator = DataCollatorForLanguageModeling(                                                                                                                              
            tokenizer=tokenizer,                                                                                                                                                      
            mlm=False,                                                                                                                                                                
        )                                                                                                                                                                             
                                                                                                                                                                                    
        trainer = Trainer(                                                                                                                                                            
            model=model,                                                                                                                                                              
            args=training_args,                                                                                                                                                       
            train_dataset=tokenized_dataset["train"],                                                                                                                                 
            eval_dataset=tokenized_dataset["validation"],                                                                                                                             
            data_collator=data_collator,                                                                                                                                              
        )                                                                                                                                                                             
                                                                                                                                                                                    
        trainer.train()                                                                                                                                                               
                                                                                                                                                                                    
        return model                                                                                                                                                                  
                                                                                                                                                                                    
    def _ptuning_fine_tuning(self, model, tokenizer, tokenized_dataset):                                                                                                              
        """Perform P-Tuning fine-tuning."""                                                                                                                                           
        # P-Tuning implementation                                                                                                                                                     
        # This is a simplified version and would need to be expanded                                                                                                                  
        # for a full implementation                                                                                                                                                   
                                                                                                                                                                                    
        from peft import PrefixTuningConfig, get_peft_model                                                                                                                           
                                                                                                                                                                                    
        ptuning_config = PrefixTuningConfig(                                                                                                                                          
            task_type=TaskType.CAUSAL_LM,                                                                                                                                             
            num_virtual_tokens=self.config.get("num_virtual_tokens", 20),                                                                                                             
            encoder_hidden_size=self.config.get("encoder_hidden_size", 128),                                                                                                          
            prefix_projection=self.config.get("prefix_projection", True),                                                                                                             
        )                                                                                                                                                                             
                                                                                                                                                                                    
        model = get_peft_model(model, ptuning_config)                                                                                                                                 
        model.print_trainable_parameters()                                                                                                                                            
                                                                                                                                                                                    
        training_args = TrainingArguments(                                                                                                                                            
            output_dir=self.config.get("output_dir", "./outputs/ptuning_fine_tuning"),                                                                                                
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
                                                                                                                                                                                    
        data_collator = DataCollatorForLanguageModeling(                                                                                                                              
            tokenizer=tokenizer,                                                                                                                                                      
            mlm=False,                                                                                                                                                                
        )                                                                                                                                                                             
                                                                                                                                                                                    
        trainer = Trainer(                                                                                                                                                            
            model=model,                                                                                                                                                              
            args=training_args,                                                                                                                                                       
            train_dataset=tokenized_dataset["train"],                                                                                                                                 
            eval_dataset=tokenized_dataset["validation"],                                                                                                                             
            data_collator=data_collator,                                                                                                                                              
        )                                                                                                                                                                             
                                                                                                                                                                                    
        trainer.train()                                                                                                                                                               
                                                                                                                                                                                    
        return model                                                                                                                                                                  
                                                                                                                                                                                    
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