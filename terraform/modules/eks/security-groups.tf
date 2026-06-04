data "aws_vpc" "this" {
  id = var.vpc_id
}

resource "aws_security_group" "cluster" {
  name        = "${var.project_name}-${var.environment}-eks-cluster"
  description = "Security group for the EKS cluster control plane"
  vpc_id      = var.vpc_id
  tags        = merge(local.common_tags, { Name = "${var.project_name}-${var.environment}-eks-cluster" })
}

resource "aws_security_group_rule" "cluster_egress" {
  type              = "egress"
  from_port         = 0
  to_port           = 0
  protocol          = "-1"
  cidr_blocks       = ["0.0.0.0/0"]
  security_group_id = aws_security_group.cluster.id
}

resource "aws_security_group_rule" "cluster_ingress_api" {
  type              = "ingress"
  from_port         = 443
  to_port           = 443
  protocol          = "tcp"
  cidr_blocks       = [data.aws_vpc.this.cidr_block]
  security_group_id = aws_security_group.cluster.id
  description       = "Allow API server access from within VPC"
}
