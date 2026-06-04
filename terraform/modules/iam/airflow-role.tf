resource "aws_iam_role" "airflow" {
  name               = local.irsa_roles.airflow.name
  assume_role_policy = data.aws_iam_policy_document.trust["airflow"].json
  tags               = local.common_tags
}

data "aws_iam_policy_document" "airflow" {
  statement {
    actions = [
      "s3:GetObject",
      "s3:PutObject",
      "s3:DeleteObject",
    ]
    resources = local.airflow_object_arns
  }

  statement {
    actions   = ["s3:ListBucket"]
    resources = local.airflow_bucket_arns
  }
}

resource "aws_iam_role_policy" "airflow" {
  name   = "${local.irsa_roles.airflow.name}-s3"
  role   = aws_iam_role.airflow.id
  policy = data.aws_iam_policy_document.airflow.json
}
