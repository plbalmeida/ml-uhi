// Provider
provider "aws" {
    region = "${var.AWS_REGION}"
}


// VPC
resource "aws_vpc" "vpc" {
    cidr_block = "10.0.0.0/16"
    enable_dns_support = "true"
    enable_dns_hostnames = "true" 
    instance_tenancy = "default"    
    
    tags = {
        Name = "VPC"
    }
}

resource "aws_subnet" "subnet-public" {
    vpc_id = "${aws_vpc.vpc.id}"
    cidr_block = "10.0.1.0/24"
    map_public_ip_on_launch = "true" // makes it a public subnet
    availability_zone = "us-east-1a"
    
    tags = {
        Name = "Public subnet"
    }
}


// Network
resource "aws_internet_gateway" "igw" {
    vpc_id = "${aws_vpc.vpc.id}"
    
    tags = {
        Name = "IGW"
    }
}

resource "aws_route_table" "rt-public" {
    vpc_id = "${aws_vpc.vpc.id}"
    
    route {
        cidr_block = "0.0.0.0/0" // associated subnet can reach everywhere 
        gateway_id = "${aws_internet_gateway.igw.id}" // RT uses this IGW to reach internet 
    }
    
    tags = {
        Name = "Route table"
    }
}

resource "aws_route_table_association" "rta-subnet-public"{
    subnet_id = "${aws_subnet.subnet-public.id}"
    route_table_id = "${aws_route_table.rt-public.id}"
}

resource "aws_security_group" "ssh-allowed" {
    vpc_id = "${aws_vpc.vpc.id}"
    
    egress {
        from_port = 0
        to_port = 0
        protocol = -1
        cidr_blocks = ["0.0.0.0/0"]
    }

    ingress {
        from_port = 22
        to_port = 22
        protocol = "tcp"
        cidr_blocks = ["0.0.0.0/0"]
    }
    
    tags = {
        Name = "SSH allowed"
    }
}


// EC2
resource "aws_instance" "ml_server" {
    ami = "ami-08c40ec9ead489470"
    # instance_type = "m5.4xlarge"
    # instance_type = "c5.24xlarge"
    instance_type = "m5.24xlarge"
    subnet_id = "${aws_subnet.subnet-public.id}"
    vpc_security_group_ids = ["${aws_security_group.ssh-allowed.id}"]
    key_name = aws_key_pair.deployer.key_name
    user_data = <<-EOF
        #!/bin/bash
        sudo apt update
        sudo apt install python3
        sudo apt install python3-pip
        sudo apt install git
    EOF
    associate_public_ip_address = true  // associar um endereço IP público

    tags = {
        Name = "MLServerInstance"
    }
}

// EIP existente
data "aws_eip" "existing" {
  id = "eipalloc-0279c9c0c4b1886a3"  // substitua pelo seu EIP ID
}

// Associa o EIP existente à instância EC2
resource "aws_eip_association" "eip_assoc" {
  instance_id = "${aws_instance.ml_server.id}"
  allocation_id = "${data.aws_eip.existing.id}"
}

// Associa o par de chaves existente à instância EC2
resource "aws_key_pair" "deployer" {
  key_name   = "ec2-key-pair"
  public_key = file("~/.ssh/ec2-key-pair.pub")
}