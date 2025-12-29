#!/bin/bash

echo "=========================================="
echo "Running All System Tests"
echo "=========================================="

cd /home/yjp/rl_dispatch_mvp || exit 1

echo ""
echo "Test 1: Industrial Safety Events & Charging Stations"
echo "------------------------------------------------------"
python test_industrial_events.py
TEST1=$?

echo ""
echo ""
echo "Test 2: Nav2 Interface & 10 Heuristic Strategies"
echo "------------------------------------------------------"
python test_nav2_and_heuristics.py
TEST2=$?

echo ""
echo ""
echo "=========================================="
echo "Test Results Summary"
echo "=========================================="

if [ $TEST1 -eq 0 ]; then
    echo "✅ Test 1: PASSED (Industrial Events)"
else
    echo "❌ Test 1: FAILED (Industrial Events)"
fi

if [ $TEST2 -eq 0 ]; then
    echo "✅ Test 2: PASSED (Nav2 & Heuristics)"
else
    echo "❌ Test 2: FAILED (Nav2 & Heuristics)"
fi

echo ""

if [ $TEST1 -eq 0 ] && [ $TEST2 -eq 0 ]; then
    echo "=========================================="
    echo "✅ ALL TESTS PASSED!"
    echo "=========================================="
    exit 0
else
    echo "=========================================="
    echo "❌ SOME TESTS FAILED"
    echo "=========================================="
    exit 1
fi
