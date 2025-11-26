#!/bin/bash
# Comprehensive Test Suite for Python Mastery Hub

echo "=========================================="
echo "Python Mastery Hub - Project Validation"
echo "=========================================="
echo ""

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

test_count=0
pass_count=0
fail_count=0

run_test() {
    local test_name="$1"
    local command="$2"
    
    test_count=$((test_count + 1))
    echo -n "Test $test_count: $test_name... "
    
    if eval "$command" > /dev/null 2>&1; then
        echo -e "${GREEN}✓ PASS${NC}"
        pass_count=$((pass_count + 1))
    else
        echo -e "${RED}✗ FAIL${NC}"
        fail_count=$((fail_count + 1))
    fi
}

# Test 1: Python version
run_test "Python version check" "python --version | grep -q '3.1'"

# Test 2: Poetry installed
run_test "Poetry installed" "poetry --version"

# Test 3: Dependencies installed  
run_test "Dependencies available" "poetry show | grep -q 'fastapi'"

# Test 4: Core module imports
run_test "Core modules import" "poetry run python -c 'from python_mastery_hub.core import get_module, list_modules'"

# Test 5: CLI module import
run_test "CLI module import" "poetry run python -c 'from python_mastery_hub.cli.main import app'"

# Test 6: All 9 learning modules load
run_test "All 9 modules load" "poetry run python -c 'from python_mastery_hub.core import list_modules; assert len(list_modules()) == 9'"

# Test 7: Module info method works
run_test "Module info retrieval" "poetry run python -c 'from python_mastery_hub.core import get_module; m = get_module(\"basics\"); assert m.get_module_info()[\"name\"]'"

# Test 8: Topics method works
run_test "Topics retrieval" "poetry run python -c 'from python_mastery_hub.core import get_module; m = get_module(\"basics\"); assert len(m.get_topics()) > 0'"

# Test 9: Demonstrate method works
run_test "Topic demonstration" "poetry run python -c 'from python_mastery_hub.core import get_module; m = get_module(\"basics\"); d = m.demonstrate(\"variables\"); assert \"examples\" in d'"

# Test 10: Learning path works
run_test "Learning path generation" "poetry run python -c 'from python_mastery_hub.core import get_learning_path; path = get_learning_path(\"beginner\"); assert len(path) > 0'"

# Test 11: CLI list command
run_test "CLI list command" "poetry run python -m python_mastery_hub.cli list-all 2>&1 | grep -q 'Python Basics'"

# Test 12: CLI path command
run_test "CLI path command" "poetry run python -m python_mastery_hub.cli path 2>&1 | grep -q 'Recommended'"

# Test 13: CLI explore command
run_test "CLI explore command" "poetry run python -m python_mastery_hub.cli explore basics 2>&1 | grep -q 'Topics'"

# Test 14: Async module working
run_test "Async programming module" "poetry run python -c 'from python_mastery_hub.core import get_module; m = get_module(\"async_programming\"); assert m.get_module_info()[\"name\"]'"

# Test 15: Config module works
run_test "Configuration module" "poetry run python -c 'from python_mastery_hub.core.config import get_settings; s = get_settings(); assert s.app_name'"

echo ""
echo "=========================================="
echo "Test Results:"
echo "=========================================="
echo -e "Total Tests:  $test_count"
echo -e "Passed:       ${GREEN}$pass_count${NC}"
echo -e "Failed:       ${RED}$fail_count${NC}"
echo ""

if [ $fail_count -eq 0 ]; then
    echo -e "${GREEN}✅ All tests passed!${NC}"
    echo ""
    echo "Python Mastery Hub is ready to use!"
    echo ""
    echo "Quick start commands:"
    echo "  CLI:"
    echo "    poetry run python -m python_mastery_hub.cli list-all"
    echo "    poetry run python -m python_mastery_hub.cli explore basics"
    echo ""
    exit 0
else
    echo -e "${RED}❌ Some tests failed${NC}"
    exit 1
fi
