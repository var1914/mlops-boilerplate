resource "aws_iam_role" "api" {
  name               = local.irsa_roles.api.name
  assume_role_policy = data.aws_iam_policy_document.trust["api"].json
  tags               = local.common_tags
}

data "aws_iam_policy_document" "api" {
  statement {
    actions = [
      "s3:GetObject",
      "s3:PutObject",
      "s3:DeleteObject",
    ]
    resources = local.data_object_arns
  }

  statement {
    actions   = ["s3:ListBucket"]
    resources = local.data_bucket_arns
  }
}

resource "aws_iam_role_policy" "api" {
  name   = "${local.irsa_roles.api.name}-s3"
  role   = aws_iam_role.api.id
  policy = data.aws_iam_policy_document.api.json
}
