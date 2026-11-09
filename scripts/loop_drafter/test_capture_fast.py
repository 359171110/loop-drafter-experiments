import unittest
from unittest.mock import patch
import torch
from .capture_protocol import capture_batch
from .test_capture_protocol import OracleTarget
from . import capture_fast as fast


class CaptureFastTests(unittest.TestCase):
    def test_append_cache_retains_storage_and_hides_future(self):
        layer=fast.AppendOnlyLayer(5)
        a=torch.arange(12,dtype=torch.float32).reshape(1,2,3,2)
        keys,values=layer.update(a,a+10)
        pointer=layer.keys.data_ptr()
        torch.testing.assert_close(keys,a)
        self.assertEqual(keys.shape[-2],3)
        b=torch.ones(1,2,1,2)
        keys,values=layer.update(b,b+10)
        torch.testing.assert_close(keys,torch.cat((a,b),dim=2))
        self.assertEqual(layer.keys.data_ptr(),pointer)
        self.assertEqual(layer.get_mask_sizes(1),(5,0))
        with self.assertRaises(ValueError): layer.update(a,a)

    def test_graph_cache_overwrites_current_position_only(self):
        layer=fast.GraphLayer(6)
        a=torch.ones(2,2,2,3)
        layer.update(a,a*2)
        layer.decode=True
        b=torch.full((2,2,1,3),7.)
        keys,_=layer.update(b,b,{'cache_position':torch.tensor([2])})
        self.assertEqual(keys.shape[-2],6)
        torch.testing.assert_close(keys[:,:,:2],a)
        torch.testing.assert_close(keys[:,:,2:3],b)
        self.assertEqual(float(keys[:,:,3:].abs().sum()),0.)
        layer.attention_length=3
        prefix,_=layer.update(b*3,b,{'cache_position':torch.tensor([2])})
        self.assertEqual(prefix.shape[-2],3)
        torch.testing.assert_close(keys[:,:,2:3],b*3)

    def test_buffered_features_match_eos_and_cap_alignment(self):
        ids=torch.tensor([[1,7],[2,8],[4,9],[9,6]])
        reference=capture_batch(OracleTarget(),ids,6,{0})
        target=OracleTarget();target.hidden_size=2
        with patch.object(fast,'CaptureTarget',return_value=target):
            actual=fast.capture_batch_fast(target,ids,6,{0})
        for (a,ah),(b,bh) in zip(reference,actual):
            self.assertTrue(torch.equal(a,b));self.assertTrue(torch.equal(ah,bh))
        self.assertEqual([len(x[0])-2 for x in actual],[1,2,4,6])
