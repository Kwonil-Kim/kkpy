from kkpy import util

def test_top_n():
    """
    make sure top_n works correctly
    """
    
    assert util.print_hello(3) == [8, 7, 4], 'incorrect'