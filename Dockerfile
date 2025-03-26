
FROM nvcr.io/nvidia/pytorch:25.01-py3                                                                                                                                                                
                                                                                                                                                                                    
WORKDIR /app                                                                                                                                                                          
                                                                                                                                                                                    
# Install system dependencies                                                                                                                                                         
RUN apt-get update && apt-get install -y \                                                                                                                                            
build-essential \                                                                                                                                                                 
git \                                                                                                                                                                             
&& rm -rf /var/lib/apt/lists/*                                                                                                                                                    
                                                                                                                                                                                    
# Copy requirements and install Python dependencies                                                                                                                                   
COPY requirements.txt .                                                                                                                                                               
RUN pip install --no-cache-dir -r requirements.txt                                                                                                                                    
                                                                                                                                                                                    
# Copy the application code                                                                                                                                                           
COPY . .                                                                                                                                                                              
                                                                                                                                                                                    
# Install the package in development mode                                                                                                                                             
RUN pip install -e .                                                                                                                                                                  
                                                                                                                                                                                    
# Set environment variables                                                                                                                                                           
ENV PYTHONPATH=/app                                                                                                                                                                   
ENV PYTHONUNBUFFERED=1                                                                                                                                                                
                                                                                                                                                                                    
# Default command                                                                                                                                                                     
ENTRYPOINT ["llm-optimizer"]                                                                                                                                                          
CMD ["--help"] 