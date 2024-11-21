# Provider Configuration
provider "aws" {
  region = var.aws_region
}

# EC2 Instance
resource "aws_instance" "llama_ec2" {
  ami           = var.ami_id
  instance_type = var.instance_type
  key_name      = var.key_name
  security_groups = [aws_security_group.llama_sg.name]
  tags = {
    Name = "Llama2-Server"
  }

  user_data = <<-EOF
              #!/bin/bash
              sudo yum update -y
              sudo yum install -y docker git
              curl -fsSL https://ollama.com/install.sh | sh
              nohup ollama serve &
              EOF
}

# Security Group
resource "aws_security_group" "llama_sg" {
  name        = "llama_sg"
  description = "Allow Flask and SSH access"

  ingress {
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]  # Allow SSH from anywhere
  }

  ingress {
    from_port   = 11434
    to_port     = 11434
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]  # Allow access to the Flask API
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
}
