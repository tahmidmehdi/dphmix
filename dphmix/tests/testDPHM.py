"""
Tahmid F. Mehdi
Moses Lab, University of Toronto
Tests
July 19, 2018

Copyright 2018 Tahmid Mehdi
This file is part of dphmix.

dphmix is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

dphmix is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with dphmix.  If not, see <http://www.gnu.org/licenses/>.
"""

from unittest import TestCase
import dphmix


class TestDPHM(TestCase):
    def test_dphm_init(self):
        dphmVar = dphmix.VariationalDPHM(alpha=1, iterations=1, max_clusters=100)
        self.assertTrue(isinstance(dphmVar, dphmix.VariationalDPHM))
        dphmGibbs = dphmix.GibbsDPHM(alpha=1, iterations=1, max_clusters=100)
        self.assertTrue(isinstance(dphmGibbs, dphmix.GibbsDPHM))
