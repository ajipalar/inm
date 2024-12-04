#!/bin/bash
grep -i $1 $2 | cut -d"," -f1-6
