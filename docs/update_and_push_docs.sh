#!/usr/bin/env bash

dest=$1
cbranch=`git symbolic-ref HEAD | sed 's!refs\/heads\/!!'`
rootdir=`pwd`

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
    make clean
    make html
    cd _build/html
    tar czf /tmp/html.tgz .
    cd ..

    if [ `pwd` == $HOME ]; then
        echo 'We are in home! Exit!'
        exit
    fi

    gh_pages=`git show-ref refs/heads/gh-pages`
    if [ -n "$gh_pages" ]; then
        git checkout gh-pages
    else
        git checkout --orphan gh-pages
    fi

    if [ `git symbolic-ref HEAD | sed 's!refs\/heads\/!!'` -ne gh-pages ]; then
	echo 'Could not switch to gh-pages!'
    fi

    git rm -rf .
    rm -rf docs/ skbold*/ img/ bin/ LICENSE *.in *.rst *.txt *.py
    tar xzf /tmp/html.tgz
    git add .
    git commit -m "Updating docs for $cbranch and pushing to origin & gh-pages branch"
    git push origin gh-pages
fi

git checkout $cbranch
