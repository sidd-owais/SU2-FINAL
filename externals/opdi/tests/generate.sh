#!/bin/bash
#
# OpDiLib, an Open Multiprocessing Differentiation Library
#
# Copyright (C) 2020-2022 Chair for Scientific Computing (SciComp), TU Kaiserslautern
# Copyright (C) 2023 Chair for Scientific Computing (SciComp), University of Kaiserslautern-Landau
# Homepage: http://www.scicomp.uni-kl.de
# Contact:  Prof. Nicolas R. Gauger (opdi@scicomp.uni-kl.de)
#
# Lead developer: Johannes Blühdorn (SciComp, University of Kaiserslautern-Landau)
#
# This file is part of OpDiLib (http://www.scicomp.uni-kl.de/software/opdi).
#
# OpDiLib is free software: you can redistribute it and/or modify it under the terms of the GNU Lesser General Public
# License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later
# version.
#
# OpDiLib is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied
# warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License for more
# details.
#
# You should have received a copy of the GNU Lesser General Public License along with OpDiLib. If not, see
# <http://www.gnu.org/licenses/>.
#

DRIVER=$1
TEST=$2
GENFILE=$BUILD_DIR"/"$DRIVER$TEST".cpp"

echo "// auto generated by OpDiLib" > $GENFILE
echo "#include \"../"$DRIVER_DIR"/Driver"$DRIVER".hpp\"" >> $GENFILE
echo "#include \"../"$TEST_DIR"/Test"$TEST".hpp\"" >> $GENFILE
echo "#include \"../case.hpp\"" >> $GENFILE
echo "" >> $GENFILE
echo "int main() {" >> $GENFILE
echo "  Case<Driver"$DRIVER", Test"$TEST">::run();" >> $GENFILE
echo "  return 0;" >> $GENFILE
echo "}" >> $GENFILE
