#ifndef PYIIJHYCRNDKCRFGIFXOFJEVULYRHGXKRNKTYCDYVSUXALPYPDKSUCVEGTXPHWYSQKPERTQIO
#define PYIIJHYCRNDKCRFGIFXOFJEVULYRHGXKRNKTYCDYVSUXALPYPDKSUCVEGTXPHWYSQKPERTQIO

#include "details/configuration.hpp"
#include "details/beams.hpp"
#include "details/tilt_matrix.hpp"
#include "details/ar_matrix.hpp"
#include "details/experimental_intensity_matrix.hpp"

namespace f
{

    struct  pattern_refinement :
            configuration< pattern_refinement >,
            beams< pattern_refinement >,
            tilt_matrix< pattern_refinement >,
            ar_matrix< pattern_refinement >,
            experimental_intenstiy_matrix< pattern_refinement >
    {
        pattern_refinement( std::string const config_path )
        {
            (*this).load_configuration( config_path );
            (*this).load_beams();
            (*this).load_tilt_matrix();
            (*this).make_ar_matrix();
            (*this).load_experimental_intensity_matrix();
            (*this).normalize_experimental_intensity_matrix();
        }
    };


}//namespace f

#endif//PYIIJHYCRNDKCRFGIFXOFJEVULYRHGXKRNKTYCDYVSUXALPYPDKSUCVEGTXPHWYSQKPERTQIO

