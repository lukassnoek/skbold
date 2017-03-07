#!/usr/bin/env bash

dest=$1
cbranch=`git symbolic-ref HEAD | sed 's!refs\/heads\/!!'`

if [ $dest == 'rtd' ]; then
    echo "Updating docs for branch '$cbranch' and pushing to origin & ReadTheDocs"
    make clean
    make html
    git add .
    git commit -m "Update docs"
    git push

else
    # Thanks to http://prjemian.github.io/gh-pages.html
    echo "Assuming serving docs from gh-pages; make docs for branch '$cbranch'"
    cd docs
    make clean
    make html
    cd build/html
    tar czf /tmp/html.tgz .
    cd ../../.. # go back to root dir

    gh_pages=`git ls-remote --heads git@github.com:lukassnoek/skbold.git gh-pages | wc -l`

    if gh_pages; then
        git checkout gh-pages
    else
        git checkout --orphan gh-pages
    fi

    git rm -rf .
    find . -path ./.git -prune -o -exec rm -rf {} \; 2> /dev/null
    tar xzf /tmp/html.tgz
    git add .
    git commit -m "Updating docs for $cbranch and pushing to origin & gh-pages branch"
    git push origin gh-pages
fi
