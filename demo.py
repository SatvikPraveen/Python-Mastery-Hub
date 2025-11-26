#!/usr/bin/env python3
"""
Python Mastery Hub - Quick Reference Guide
Interactive demonstration of project features
"""

from python_mastery_hub.core import get_module, list_modules, get_learning_path

def print_section(title):
    """Print formatted section header."""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}\n")

def demo_all_modules():
    """Demonstrate all available modules."""
    print_section("1. ALL AVAILABLE MODULES")
    
    modules = list_modules()
    for i, module in enumerate(modules, 1):
        print(f"{i:2}. {module['name']:30} ({module['difficulty'].title():12})")
        print(f"    {module['description'][:60]}...")
    
    print(f"\nTotal: {len(modules)} modules")

def demo_module_details():
    """Show details of a specific module."""
    print_section("2. MODULE DETAILS - PYTHON BASICS")
    
    module = get_module("basics")
    info = module.get_module_info()
    
    print(f"Name:         {info['name']}")
    print(f"Difficulty:   {info['difficulty'].title()}")
    print(f"Description:  {info['description']}")
    
    topics = module.get_topics()
    print(f"\nTopics ({len(topics)}):")
    for i, topic in enumerate(topics, 1):
        print(f"  {i}. {topic.replace('_', ' ').title()}")

def demo_topic_example():
    """Show example for a specific topic."""
    print_section("3. TOPIC EXAMPLE - VARIABLES")
    
    module = get_module("basics")
    demo_data = module.demonstrate("variables")
    
    print(f"Explanation: {demo_data['explanation']}\n")
    
    if demo_data.get('examples'):
        print("Available Examples:")
        for example_name in list(demo_data['examples'].keys())[:2]:
            print(f"  • {example_name.replace('_', ' ').title()}")

def demo_learning_paths():
    """Show learning paths."""
    print_section("4. LEARNING PATHS")
    
    for difficulty in ["beginner", "intermediate", "advanced"]:
        path = get_learning_path(difficulty)
        print(f"{difficulty.title()} Path ({len(path)} modules):")
        for i, module_name in enumerate(path, 1):
            module = get_module(module_name)
            print(f"  {i}. {module.get_module_info()['name']}")
        print()

def demo_cli_commands():
    """Show available CLI commands."""
    print_section("5. CLI COMMANDS")
    
    commands = [
        ("list-all", "Show all learning modules", 
         "poetry run python -m python_mastery_hub.cli list-all"),
        ("path", "Get recommended learning path",
         "poetry run python -m python_mastery_hub.cli path --difficulty beginner"),
        ("explore", "Explore a specific module",
         "poetry run python -m python_mastery_hub.cli explore basics"),
        ("info", "Show platform information",
         "poetry run python -m python_mastery_hub.cli info"),
    ]
    
    for cmd, desc, example in commands:
        print(f"Command: {cmd}")
        print(f"  Description: {desc}")
        print(f"  Example:     {example}\n")

def main():
    """Run all demonstrations."""
    print("\n" + "="*60)
    print("  PYTHON MASTERY HUB - QUICK REFERENCE")
    print("="*60)
    
    try:
        demo_all_modules()
        demo_module_details()
        demo_topic_example()
        demo_learning_paths()
        demo_cli_commands()
        
        print_section("VALIDATION COMPLETE")
        print("✅ All core functionality working correctly!")
        print("\nNext Steps:")
        print("  1. Try: poetry run python -m python_mastery_hub.cli list-all")
        print("  2. Explore: poetry run python -m python_mastery_hub.cli explore basics")
        print("  3. Learn: Follow the recommended learning path")
        print("\nHappy Learning! 🐍\n")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
