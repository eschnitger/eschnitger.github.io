#!/bin/zsh
# Easy blog posting for the Writings section.
#
#   ./post.sh new "My Post Title"   create writing/YYYY-MM-DD-my-post-title/index.qmd and open it
#   ./post.sh preview               live preview of the site in your browser
#   ./post.sh publish               commit the post and publish the site to gh-pages (goes live)

set -e
cd "$(dirname "$0")"

case "$1" in
  new)
    title="$2"
    [[ -z "$title" ]] && { echo 'Usage: ./post.sh new "Post Title"'; exit 1; }
    slug=$(echo "$title" | tr '[:upper:]' '[:lower:]' | sed -E 's/[^a-z0-9]+/-/g; s/^-+|-+$//g')
    today=$(date +%Y-%m-%d)
    dir="writing/$today-$slug"
    file="$dir/index.qmd"
    [[ -e "$file" ]] && { echo "Already exists: $file"; exit 1; }
    mkdir -p "$dir"
    cat > "$file" <<EOF
---
title: "$(printf '%s' "$title" | sed 's/"/\\"/g')"
description: ""
author: "Eric Schnitger"
date: "$today"
categories: []
draft: false
---

Start writing here.
EOF
    echo "Created $file"
    if [[ -z "$NO_OPEN" ]]; then open "$file"; fi
    ;;

  preview)
    quarto preview
    ;;

  publish)
    # Save the post source to main, then build the site and push it to gh-pages (what GitHub Pages serves).
    git add writing post.sh
    if ! git diff --cached --quiet; then
      git commit -m "${2:-Update writings}"
      git push
    fi
    quarto publish gh-pages --no-prompt --no-browser
    echo "Published. Live in a minute or two at https://eschnitger.github.io/writing/"
    ;;

  *)
    sed -n '2,7p' "$0" | sed 's/^# \{0,1\}//'
    ;;
esac
