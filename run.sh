#!/bin/bash

git pull origin main
sudo docker build . -t fyp && sudo docker run -p 7000:7000 fyp