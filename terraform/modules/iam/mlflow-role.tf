resource "aws_iam_role" "mlflow" {
  name               = local.irsa_roles.mlflow.name
  assume_role_policy = data.aws_iam_policy_document.trust["mlflow"].json
  tags               = local.common_tags
}

data "aws_iam_policy_document" "mlflow" {
  statement {
    actions = [
      "s3:GetObject",
      "s3:PutObject",
      "s3:DeleteObject",
    ]
    resources = ["${var.mlflow_artifacts_bucket_arn}/*"]
  }

  statement {
    actions   = ["s3:ListBucket"]
    resources = [var.mlflow_artifacts_bucket_arn]
  }
}

resource "aws_iam_role_policy" "mlflow" {
  name   = "${local.irsa_roles.mlflow.name}-s3"
  role   = aws_iam_role.mlflow.id
  policy = data.aws_iam_policy_document.mlflow.json
}
