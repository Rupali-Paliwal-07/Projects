try:
    with open("test_file.txt", "w") as test_file:
        test_file.write("This is a test.")
    print("Test file created successfully.")
except IOError as e:
    print(f"Error creating test file: {e}")
