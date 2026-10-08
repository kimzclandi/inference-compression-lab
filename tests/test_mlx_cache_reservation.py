import unittest
from lab.mlx_cache_reservation import reserve_fresh_caches

class Native:
    def __init__(self):
        self.offset=0; self.keys=None; self.values=None; self.step=256

class ReservationTests(unittest.TestCase):
    def test_fresh_request(self):
        caches=[Native(),Native()]
        self.assertIs(reserve_fresh_caches(caches,288,Native),caches)
        self.assertEqual([c.step for c in caches],[288,288])
    def test_validation_is_before_mutation(self):
        caches=[Native(),Native()]; caches[1].offset=1
        with self.assertRaises(ValueError): reserve_fresh_caches(caches,288,Native)
        self.assertEqual(caches[0].step,256)
    def test_bad_capacity_and_cache(self):
        for n in [0,-1,True,1.5]:
            with self.assertRaises(ValueError): reserve_fresh_caches([Native()],n,Native)
        with self.assertRaises(ValueError): reserve_fresh_caches([],1,Native)
        class Derived(Native):pass
        with self.assertRaises(ValueError): reserve_fresh_caches([Derived()],1,Native)
        for field,value in [('keys',object()),('values',object()),('step',128)]:
            cache=Native(); setattr(cache,field,value)
            with self.assertRaises(ValueError): reserve_fresh_caches([cache],288,Native)
