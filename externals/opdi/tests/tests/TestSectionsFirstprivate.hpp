﻿/*
 * OpDiLib, an Open Multiprocessing Differentiation Library
 *
 * Copyright (C) 2020-2022 Chair for Scientific Computing (SciComp), TU Kaiserslautern
 * Copyright (C) 2023 Chair for Scientific Computing (SciComp), University of Kaiserslautern-Landau
 * Homepage: http://www.scicomp.uni-kl.de
 * Contact:  Prof. Nicolas R. Gauger (opdi@scicomp.uni-kl.de)
 *
 * Lead developer: Johannes Blühdorn (SciComp, University of Kaiserslautern-Landau)
 *
 * This file is part of OpDiLib (http://www.scicomp.uni-kl.de/software/opdi).
 *
 * OpDiLib is free software: you can redistribute it and/or modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later
 * version.
 *
 * OpDiLib is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied
 * warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License for more
 * details.
 *
 * You should have received a copy of the GNU Lesser General Public License along with OpDiLib. If not, see
 * <http://www.gnu.org/licenses/>.
 *
 */

#pragma once

#include "testBase.hpp"

template<typename _Case>
struct TestSectionsFirstprivate : public TestBase<4, 1, 3, TestSectionsFirstprivate<_Case>> {
  public:
    using Case = _Case;
    using Base = TestBase<4, 1, 3, TestSectionsFirstprivate<Case>>;

    template<typename T>
    static void test(std::array<T, Base::nIn> const& in, std::array<T, Base::nOut>& out) {

      int const N = 1000;
      T* jobResults = new T[N];

      T helper = in[0] * in[1] * in[2] * in[3];

      OPDI_PARALLEL()
      {
        OPDI_SECTIONS(firstprivate(helper))
        {
          OPDI_SECTION()
          {
            helper = cos(helper);

            for (int i = 0; i < N / 3; ++i) {
              Base::job1(i, in, jobResults[i]);
              jobResults[i] *= helper;
            }

            helper = in[0] * in[1] * in[2] * in[3];
          }
          OPDI_END_SECTION

          OPDI_SECTION()
          {
            helper = sin(helper);

            for (int i = N / 3; i < 2 * (N / 3); ++i) {
              Base::job1(i, in, jobResults[i]);
              jobResults[i] *= helper;
            }

            helper = in[0] * in[1] * in[2] * in[3];
          }
          OPDI_END_SECTION

          OPDI_SECTION()
          {
            helper = sin(cos(helper));

            for (int i = 2 * (N / 3); i < N; ++i) {
              Base::job1(i, in, jobResults[i]);
              jobResults[i] *= helper;
            }
          }
          OPDI_END_SECTION
        }
        OPDI_END_SECTIONS
      }
      OPDI_END_PARALLEL

      for (int i = 0; i < N; ++i) {
        out[0] += jobResults[i];
      }

      delete [] jobResults;
    }
};