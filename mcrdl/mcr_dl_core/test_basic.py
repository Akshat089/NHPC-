import mcrdl

def test_hello():
    c = mcrdl.Comm()
    c.say_hello()

if __name__ == "__main__":
    test_hello()
    print("Test passed")
