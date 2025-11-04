import mcrdl

def test_hello():
    c = mcrdl.Comm()
    c.init()
    c.all_to_all()

if __name__ == "__main__":
    test_hello()
    print("Test passed")
