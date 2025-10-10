from grid_feedback_optimizer.main import main

def test_main_simple_example():
    # Example usage: user edits this line with their JSON path
    res = main("./examples/simple_example.json")
    res.save(output_file='tests/output/optimized_simple_example.json')

def test_main_simple_example_with_transformer():
    # Example usage: user edits this line with their JSON path
    res = main("./examples/simple_example_with_transformer.json")
    res.save(output_file='tests/output/optimized_simple_example_with_transformer.json')
