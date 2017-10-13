#ifndef QRFDPLRDNFLTYTROSVHHAEDWETDFWRVTLEFHFLVXEGGCVNBHWUOYUKAWFAXHFGGOAFNWWEMLY
#define QRFDPLRDNFLTYTROSVHHAEDWETDFWRVTLEFHFLVXEGGCVNBHWUOYUKAWFAXHFGGOAFNWWEMLY

#include <atomic>
#include <condition_variable>
#include <deque>
#include <memory> //allocator
#include <mutex>
#include <thread>
#include <utility>
#include <vector>

namespace f
{
    template< typename Product_Type >
    struct repository
    {
        typedef unsigned long                                       index_type;
        typedef unsigned long                                       size_type;
        typedef Product_Type                                        product_type;
        typedef std::deque<product_type>                            product_container_type;
        typedef std::thread                                         thread_type;
        typedef std::vector<thread_type>                            consumer_container_type;
        typedef std::vector<thread_type>                            producer_container_type;
        typedef std::mutex                                          mutex_type;
        typedef std::atomic_bool                                    bool_type;
        typedef std::condition_variable                             condition_variable;

        size_type                                                   product_capacity;

        //product repository
        product_container_type                                      product_package;
        bool_type                                                   product_ready;
        mutex_type                                                  product_mutex;
        condition_variable                                          product_condition;
        //
        consumer_container_type                                     consumer_package;
        mutex_type                                                  consumer_mutex;
        //
        producer_container_type                                     producer_package;
        mutex_type                                                  producer_mutex;

        void put_new_product( product_type const& product )
        {
            {
                std::unique_lock<std::mutex> lk{ product_mutex };
                product_condition.wait( lk, [this](){ return (*this).product_package.size() <= (*this).product_capacity; } );
                product_package.push_front( product );
                product_ready = true;
            }
            product_condition.notify_all();
        }

        product_type const get_new_product()
        {
            //TODO:
            //specialization -- is_default_constructible and is_move_constructable
            std::unique_lock<std::mutex> lk{ product_mutex };
            product_condition.wait( lk, [this](){ return (*this).product_ready == true; } );
            product_type product{ std::move(product_package.back()) };
            product_package.pop_back();
            if ( product_package.empty() )
                product_ready = false;
            product_condition.notify_all();
            return product;
        }

        template< typename Producer >
        void add_producer( Producer const& producer )
        {
            std::lock_guard<std::mutex> lg{ producer_mutex };
            producer_package.emplace_back( producer, [this](product_type const& pd ){ (*this).put_new_product(pd); } );
        }

        template< typename Consumer >
        void add_consumer( Consumer const& consumer )
        {
            std::lock_guard<std::mutex> lg{ consumer_mutex };
            consumer_package.emplace_back( consumer, [this](){ return (*this).get_new_product(); } );
        }

        repository( size_type product_capacity_ = 256 ) : product_capacity( product_capacity_ ), product_ready( false ) {} 

        ~repository()
        {
            for ( auto&& producer : producer_package )
                if ( producer.joinable() ) 
                    producer.join();

            for ( auto&& consumer : consumer_package )
                if ( consumer.joinable() ) 
                    consumer.join();
        }

    };//struct repository

}//namespace f

#endif//QRFDPLRDNFLTYTROSVHHAEDWETDFWRVTLEFHFLVXEGGCVNBHWUOYUKAWFAXHFGGOAFNWWEMLY

