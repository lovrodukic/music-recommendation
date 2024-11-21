output "instance_public_ip" {
  description = "Public IP address of the EC2 instance"
  value       = aws_instance.llama_ec2.public_ip
}
