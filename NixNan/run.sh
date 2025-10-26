#!/bin/bash
while true; do
    echo -e "\n1) Install  2) Priority  3) Analyze  4) nixnan-priority  0) Exit"
    read -p "Choose: " c
    case $c in
        1) ./install.sh ;;
        2) make priority ;;
        3) make analyze ;;
        4) make nixnan-priority ;;
        0) exit 0 ;;
    esac
done
