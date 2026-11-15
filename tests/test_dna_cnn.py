import pytest
import torch
from dna_cnn import DNACNN, create_dna_cnn

class TestDNACNN:
    def test_init(self):
        m = DNACNN(dim=256, hidden_dim=512)
        assert m.dim == 256
    
    def test_forward(self):
        m = DNACNN(dim=256)
        x = torch.randn(32, 256)
        assert m(x).shape == (32, 256)
    
    def test_factory(self):
        m = create_dna_cnn()
        assert isinstance(m, DNACNN)

if __name__ == '__main__':
    pytest.main([__file__, '-v'])
