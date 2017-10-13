#ifndef MLOGGER_HPP_SDFPONSAFLKJSANDFLKASFDUIOH439Y8UHASFDKLJAHSFDLKJSAHFDLKAJFD
#define MLOGGER_HPP_SDFPONSAFLKJSANDFLKASFDUIOH439Y8UHASFDKLJAHSFDLKJSAHFDLKAJFD

#include <string>
#include <ctime>
#include <fstream>
#include <mutex>
#include <iomanip>
#include <memory>
#include <sstream>

namespace f
{

template<typename Detailed_Log_Policy>
struct log_policy
{
    typedef Detailed_Log_Policy zen_type;

    void open( const std::string& name )
    {
        auto& zen = static_cast<zen_type&>(*this);
        zen.open_impl( name );
    }

    void close()
    {
        auto& zen = static_cast<zen_type&>(*this);
        zen.close_impl( );
    }

    void write( const std::string& msg )
    {
        auto& zen = static_cast<zen_type&>(*this);
        zen.write_impl( msg );
    }
};//log_policy

struct file_log_policy : log_policy< file_log_policy >
{
    std::unique_ptr< std::ofstream > out_stream;

    file_log_policy() : out_stream ( new std::ofstream ) {}

    void open_impl( const std::string& name )
    {
        (*out_stream).open( name.c_str(), std::ios_base::binary | std::ios_base::out );

        if ( !((*out_stream).is_open()) )
            throw( std::runtime_error( "LOGGER: Unable to open an output stream" ) );
    }

    void close_impl()
    {
        if ( out_stream )
            (*out_stream).close();
    }

    void write_impl( const std::string& msg )
    {
        ( *out_stream ) << msg << std::endl;
    }

    ~file_log_policy()
    {
        if( out_stream )
            (*out_stream).close();
    }
};//file_log_policy

enum severity_type
{
    debug = 1,
    error,
    warning
};

template< typename log_policy >
struct logger
{
    unsigned                log_line_number;
    std::stringstream       log_stream;
    log_policy*             policy;
    std::mutex              write_mutex;

    std::string get_time()
    {
        std::string     time_str;
        time_t          raw_time;

        time( &raw_time );
        time_str = ctime( &raw_time );

        return time_str.substr( 0 , time_str.size() - 1 );
    }

    std::string get_logline_header()
    {
        std::stringstream header;

        header.str( "" );
        header.fill( '0' );
        header.width( 7 );
        header << log_line_number++ << " < " << get_time() << " - ";

        header.fill( '0' );
        header.width( 7 );
        header << clock() << " > ~ ";

        return header.str();
    }

    void print_impl()
    {
        (*policy).write( get_logline_header() + log_stream.str() );
        log_stream.str( "" );
    }

    template<typename First, typename...Rest>
    void print_impl( First parm1, Rest...parm )
    {
        log_stream << parm1;
        print_impl( parm... );
    }

    logger( const std::string& name ) : log_line_number(0), policy( new log_policy )
    {
        if ( !policy )
            throw std::runtime_error ( "LOGGER: Unable to create the logger instance" );

        policy->open( name );
    }

    template< severity_type severity , typename...Args >
    void print( Args...args )
    {
        std::lock_guard<std::mutex> lk( write_mutex );

        switch ( severity )
        {
            case severity_type::debug:
                log_stream << "<DEBUG> :";
                break;

            case severity_type::warning:
                log_stream << "<WARNING> :";
                break;

            case severity_type::error:
                log_stream << "<ERROR> :";
                break;
        };

        print_impl( args... );
    }

    ~logger()
    {
        if ( policy )
        {
            (*policy).close();
            delete policy;
        }
    }
};

} //namespace f

static f::logger< f::file_log_policy > log_inst( "execution.log" );

#endif//_LOGGER_HPP_SDFPONSAFLKJSANDFLKASFDUIOH439Y8UHASFDKLJAHSFDLKJSAHFDLKAJFD

#ifdef LOG_DEBUG
#undef LOG_DEBUG
#endif

#ifdef LOG_ERROR
#undef LOG_ERROR
#endif

#ifdef LOG_WARNING
#undef LOG_WARNING
#endif

#ifdef DEBUG
#define LOG_DEBUG log_inst.print< f::severity_type::debug >
#define LOG_ERROR log_inst.print< f::severity_type::error >
#define LOG_WARNING log_inst.print< f::severity_type::warning >
#else
#define LOG_DEBUG(...)
#define LOG_ERROR(...)
#define LOG_WARNING(...)
#endif

/*
#ifdef LOGGING_LEVEL_2
#define ELOG_DEBUG log_inst.print< f::severity_type::debug >
#define ELOG_ERROR log_inst.print< f::severity_type::error >
#define ELOG_WARNING log_inst.print< f::severity_type::warning >
#else
#define ELOG_DEBUG(...)
#define ELOG_ERROR(...)
#define ELOG_WARNING(...)
#endif
*/


