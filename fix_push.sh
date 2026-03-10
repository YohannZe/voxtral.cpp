#!/bin/bash
git credential reject <<EOF
protocol=https
host=github.com
EOF

git push origin main
