#ifndef IEOWQPVLNHTDMAWCDMWGMGNVEECYNKPYSQBFGOQWRXDKXNQJYNBRDVPIKWULIHUPOVYCBJKTN
#define IEOWQPVLNHTDMAWCDMWGMGNVEECYNKPYSQBFGOQWRXDKXNQJYNBRDVPIKWULIHUPOVYCBJKTN

#include <f/singleton/singleton.hpp>

#include <cmath>
#include <string>
#include <sstream>
#include <vector>
#include <iostream>
#include <cstddef>

namespace test_private_sdioaslkdfjn43poiuhaflkjhasfdlkjhasfdoiuhlahflsfdhaskldfjh
{
    typedef std::vector<std::string> test_result_type;

}//namespace test_[test_private_sdioaslkdfjn43poiuhaflkjhasfdlkjhasfdoiuhlahflsfdhaskldfjh

namespace f
{
    struct test
    {
        //typedef test_private_sdioaslkdfjn43poiuhaflkjhasfdlkjhasfdoiuhlahflsfdhaskldfjh::test_result_type test_result_type;
        typedef std::vector<std::string> test_result_type;

        test( const std::string& test_name_ ) : test_name( test_name_ ) {}

        virtual ~test() {}

        virtual void run( test_result_type& result ) = 0;

        void record_failure( test_result_type& result, const std::string& file, unsigned long int line, const std::string& message )
        {
            // If the full filename is too long, only use the last part.
            std::string file_string = file;
            std::size_t max_length = 140;
            std::size_t file_length = file.size();
            if ( file_length > max_length )
            {
                // Get the last max_length characters - 3 (leave room for
                // three ellipses at the beginning).
                file_string = "...";
                file_string += file.substr( file_length - max_length + 3, file_length - 1 );
            }
            std::ostringstream oss;
            oss << file_string << "(" << line << "): '" << test_name << "' FAILED: " << message;
            result.push_back( oss.str() );
        }

        /// The unique name of this test.
        std::string test_name;
    };

    struct test_manager
    {
        //typedef test_private_sdioaslkdfjn43poiuhaflkjhasfdlkjhasfdoiuhlahflsfdhaskldfjh::test_result_type test_result_type;
        typedef std::vector<std::string> test_result_type;

        void add_test( test* test )
        {
            all_tests.push_back( test );
        }

        void run_tests()
        {
            if ( !all_test_results.size() ) return;

            std::size_t total_failures = 0;
            *output_stream << "[-------------- RUNNING UNIT TESTS --------------]" << std::endl;
            std::vector<test*>::iterator iter;
            for ( iter = all_tests.begin(); iter != all_tests.end(); ++iter )
            {
                ( *iter )->run( all_test_results );
                bool testFailed = false;
                size_t size = all_test_results.size();
                for ( size_t i = 0; i < size; ++i )
                {
                    *output_stream << all_test_results.at( i ) << std::endl;
                    testFailed = true;
                }
                all_test_results.clear();
                if ( testFailed )
                {
                    ++total_failures;
                }
            }
            *output_stream << "Results: " << all_tests.size() - total_failures << " succeeded, " << total_failures << " failed" << std::endl;
            *output_stream << "[-------------- UNIT TESTS FINISHED -------------]" << std::endl;

            all_test_results.clear();
        }

        test_manager()
        {
            output_stream = &std::cout;
        }

        ~test_manager()
        {
            run_tests();
        }

        std::vector<test*> all_tests;

        std::ostream* output_stream;

        test_result_type all_test_results;
    };
}//namespace f

#ifdef TEST
#undef TEST
#endif
#define TEST(test_name)\
    class test_name##_test : public f::test\
    {\
        public:\
            test_name##_test()\
                : test(#test_name)\
            {\
                auto& manager = f::singleton<f::test_manager>::instance(); \
                manager.add_test(this); \
            }\
            void run(test_private_sdioaslkdfjn43poiuhaflkjhasfdlkjhasfdoiuhlahflsfdhaskldfjh::test_result_type& result);\
    };\
    test_name##_test test_name##_instance;\
    void test_name##_test::run(test_private_sdioaslkdfjn43poiuhaflkjhasfdlkjhasfdoiuhlahflsfdhaskldfjh::test_result_type& result)

#ifdef RUN_TESTS
#undef RUN_TESTS
#endif
#define RUN_TESTS  \
    auto& manager = f::singleton<f::test_manager>::instance(); \
    manager.run_tests()

#ifdef SET_OUTPUT
#undef SET_OUTPUT
#endif
#define SET_OUTPUT(stream)\
    auto& manager = f::singleton<f::test_manager>::instance(); \
    manager.output_stream = stream;

#ifdef CHECK
#undef CHECK
#endif
#define CHECK(condition)\
    {\
        if (!(condition))\
        {\
            record_failure(result, __FILE__, __LINE__, #condition);\
        }\
    }

#ifdef CHECK_EQUAL
#undef CHECK_EQUAL
#endif
#define CHECK_EQUAL(value1, value2)\
    {\
        if ((value1) != (value2))\
        {\
            std::ostringstream oss;\
            oss << "value1 (" << (value1) << ") should equal "\
                << "value2 (" << (value2) << ")";\
            record_failure(result, __FILE__, __LINE__, oss.str());\
        }\
    }

#ifdef CHECK_NOT_EQUAL
#undef CHECK_NOT_EQUAL
#endif
#define CHECK_NOT_EQUAL(value1, value2)\
    {\
        if ((value1) == (value2))\
        {\
            std::ostringstream oss;\
            oss << "value1 (" << (value1) << ") should not equal "\
                << "value2 (" << (value2) << ")";\
            record_failure(result, __FILE__, __LINE__, oss.str());\
        }\
    }

#ifdef CHECK_CLOSE
#undef CHECK_CLOSE
#endif
#define CHECK_CLOSE(value1, value2, tolerance)\
    {\
        double tempValue1 = (double)(value1);\
        double tempValue2 = (double)(value2);\
        if (std::abs((tempValue1)-(tempValue2)) > tolerance)\
        {\
            std::ostringstream oss;\
            oss << "value1 (" << (value1) << ") should be close to "\
                << "value2 (" << (value2) << ")";\
            record_failure(result, __FILE__, __LINE__, oss.str());\
        }\
    }

#ifdef CHECK_LESS
#undef CHECK_LESS
#endif
#define CHECK_LESS(value1, value2)\
    {\
        if ((value1) >= (value2))\
        {\
            std::ostringstream oss;\
            oss << "value1 (" << (value1) << ") should be less than "\
                << "value2 (" << (value2) << ")";\
            record_failure(result, __FILE__, __LINE__, oss.str());\
        }\
    }

#ifdef CHECK_LESS_OR_EQUAL
#undef CHECK_LESS_OR_EQUAL
#endif
#define CHECK_LESS_OR_EQUAL(value1, value2)\
    {\
        if ((value1) > (value2))\
        {\
            std::ostringstream oss;\
            oss << "value1 (" << (value1) << ") should be less than or "\
                << "equal to " << "value2 (" << (value2) << ")";\
            record_failure(result, __FILE__, __LINE__, oss.str());\
        }\
    }


#ifdef CHECK_GREATER
#undef CHECK_GREATER
#endif
#define CHECK_GREATER(value1, value2)\
    {\
        if ((value1) <= (value2))\
        {\
            std::ostringstream oss;\
            oss << "value1 (" << (value1) << ") should be greater than "\
                << "value2 (" << (value2) << ")";\
            record_failure(result, __FILE__, __LINE__, oss.str());\
        }\
    }

#ifdef CHECK_GREATER_OR_EQUAL
#undef CHECK_GREATER_OR_EQUAL
#endif
#define CHECK_GREATER_OR_EQUAL(value1, value2)\
    {\
        if ((value1) < (value2))\
        {\
            std::ostringstream oss;\
            oss << "value1 (" << (value1) << ") should be greater than or "\
                << "equal to " << "value2 (" << (value2) << ")";\
            record_failure(result, __FILE__, __LINE__, oss.str());\
        }\
    }



#ifdef FAIL
#undef FAIL
#endif
#define FAIL(message)\
    {\
        record_failure(result, __FILE__, __LINE__, (message));\
    }\

#ifdef PRINT
#undef PRINT
#endif
#define PRINT(message)\
    {\
        auto& manager = f::singleton<test_manager>::instance(); \
        (*(manager.output_stream)) << (message) << std::flush; \
    }\

#endif//IEOWQPVLNHTDMAWCDMWGMGNVEECYNKPYSQBFGOQWRXDKXNQJYNBRDVPIKWULIHUPOVYCBJKTN

