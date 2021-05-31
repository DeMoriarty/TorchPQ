import unittest

class CustomTestCase(unittest.TestCase):
  def assertTensorEqual(self, a, b):
    self.assertTrue( (a != b).sum().item() == 0 )
  
  def assertTensorNotEqual(self, a, b):
    self.assertFalse( (a != b).sum().item() == 0 )

  def assertErrorLessThan(self, a, b, threshold=0, norm="l2"):
    if norm == "l2":
      dif = (a - b).pow(2)
    elif norm == "l1":
      dif = (a - b).abs()
    self.assertTrue(dif.sum().item() <= threshold)
  
  def assertErrorNotLessThan(self, a, b, threshold=0, norm="l2"):
    if norm == "l2":
      dif = (a - b).pow(2)
    elif norm == "l1":
      dif = (a - b).abs()
    self.assertFalse(dif.sum().item() <= threshold)
