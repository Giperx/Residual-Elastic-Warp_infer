conda env list | awk '{print $1}' | grep -v "^#" | while read env; do
  echo "==== 环境：$env ===="
  conda list -n "$env" | grep -E "pytorch|torchvision|torchaudio"
done