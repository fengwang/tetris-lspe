namespace tcl
{
    template<typename T, typename U, typename V> class basic_tree;
}

template< typename stored_type, typename tree_type, typename container_type >
class tcl::basic_tree
{
    public:

        typedef basic_tree<stored_type, tree_type, container_type> basic_tree_type;
        typedef stored_type* ( *tClone_fcn )( const stored_type& );
        typedef stored_type value_type;
        typedef stored_type& reference;
        typedef const stored_type& const_reference;
        typedef size_t size_type;
        typedef std::allocator<stored_type> allocator_type;
        typedef typename allocator_type::difference_type difference_type;

    protected:

        basic_tree() : pElement( 0 ), pParent_node( 0 ) {}
        explicit basic_tree( const stored_type& value );
        basic_tree( const basic_tree_type& rhs );
        virtual ~basic_tree();

    public:
        const stored_type* get() const { return pElement;}
        stored_type* get() { return pElement;}
        bool is_root() const { return pParent_node == 0;}
        size_type size() const { return children.size();}
        size_type max_size() const { return( std::numeric_limits<int>().max )();}
        bool empty() const { return children.empty();}
        tree_type* parent() { return pParent_node;}
        const tree_type* parent() const { return pParent_node;}
        static void set_clone( const tClone_fcn& fcn ) { pClone_fcn = fcn;}

    protected:
        void set_parent( tree_type* pParent ) { pParent_node = pParent;}
        basic_tree_type& operator = ( const basic_tree_type& rhs );
        void set( const stored_type& stored_obj );
        void allocate_stored_type( stored_type*& element_ptr, const stored_type& value )
        { element_ptr = stored_type_allocator.allocate( 1, 0 ); stored_type_allocator.construct( element_ptr, value );}
        void deallocate_stored_type( stored_type* element_ptr )
        { stored_type_allocator.destroy( element_ptr ); stored_type_allocator.deallocate( element_ptr, 1 );}
        void allocate_tree_type( tree_type*& tree_ptr, const tree_type& tree_obj )
        { tree_ptr = tree_type_allocator.allocate( 1, 0 ); tree_type_allocator.construct( tree_ptr, tree_obj );}
        void deallocate_tree_type( tree_type* tree_ptr )
        { tree_type_allocator.destroy( tree_ptr ); tree_type_allocator.deallocate( tree_ptr, 1 );}


    protected:
        container_type children;
    private:
        stored_type* pElement;
        mutable tree_type* pParent_node;
        static tClone_fcn pClone_fcn;
        std::allocator<stored_type> stored_type_allocator;
        std::allocator<tree_type> tree_type_allocator;
};

// 1 "basic_tree.inl" 1
// 29 "basic_tree.inl"
template< typename stored_type, typename tree_type, typename container_type >
typename tcl::basic_tree<stored_type, tree_type, container_type>::tClone_fcn
tcl::basic_tree<stored_type, tree_type, container_type>::pClone_fcn = 0;


template< typename stored_type, typename tree_type, typename container_type >
tcl::basic_tree<stored_type, tree_type, container_type>::basic_tree( const stored_type& value )
    : children( container_type() ), pElement( 0 ), pParent_node( 0 ),
      stored_type_allocator( std::allocator<stored_type>() ), tree_type_allocator( std::allocator<tree_type>() )
{
    if ( pClone_fcn )
    { pElement = pClone_fcn( value ); }

    else
    { allocate_stored_type( pElement, value ); }
}



template< typename stored_type, typename tree_type, typename container_type >
tcl::basic_tree<stored_type, tree_type, container_type>::basic_tree( const basic_tree_type& rhs )
    : children( container_type() ), pElement( 0 ), pParent_node( 0 ),
      stored_type_allocator( std::allocator<stored_type>() ), tree_type_allocator( std::allocator<tree_type>() )
{
    pParent_node = 0;
    set( *rhs.get() );
}


template< typename stored_type, typename tree_type, typename container_type >
tcl::basic_tree<stored_type, tree_type, container_type>& tcl::basic_tree<stored_type, tree_type, container_type>::operator = ( const basic_tree_type& rhs )
{
    if ( &rhs == this )
    { return *this; }

    set( *rhs.get() );
    return *this;
}


template< typename stored_type, typename tree_type, typename container_type >
tcl::basic_tree<stored_type, tree_type, container_type>::~basic_tree()
{
    deallocate_stored_type( pElement );
}




template< typename stored_type, typename tree_type, typename container_type >
void tcl::basic_tree<stored_type, tree_type, container_type>::set( const stored_type& value )
{
    if ( pElement )
    { deallocate_stored_type( pElement ); }

    if ( pClone_fcn )
    { pElement = pClone_fcn( value ); }

    else
    { allocate_stored_type( pElement, value ); }
}

namespace tcl
{

    template<typename T, typename U, typename V> class basic_tree;
    template<typename T> class sequential_tree;
    template<typename T, typename U, typename V> class associative_tree;
    template<typename T, typename U, typename V, typename W, typename X, typename Y> class associative_reverse_iterator;
    template<typename T, typename U, typename V, typename W, typename X, typename Y> class sequential_reverse_iterator;
    template<typename T, typename U, typename V, typename W, typename X, typename Y, typename Z> class pre_order_descendant_iterator;
    template<typename T, typename U, typename V, typename W, typename X, typename Y, typename Z> class post_order_descendant_iterator;

    template<typename T, typename U, typename V, typename W, typename X, typename Y> class associative_iterator;
    template<typename T, typename U, typename V, typename W, typename X, typename Y> class sequential_iterator;

    template<typename ST, typename TT, typename TPT1, typename CT, typename PT1, typename RT1, typename TPT2, typename PT2, typename RT2>
    void associative_it_init( associative_iterator<ST, TT, TPT1, CT, PT1, RT1>* dest, const associative_iterator<ST, TT, TPT2, CT, PT2, RT2>& src ) { dest->it = src.it; dest->pParent = src.pParent; }

    template<typename ST, typename TT, typename TPT1, typename CT, typename PT1, typename RT1, typename TPT2, typename PT2, typename RT2>
    bool associative_it_eq( const associative_iterator<ST, TT, TPT1, CT, PT1, RT1>* lhs, const associative_iterator<ST, TT, TPT2, CT, PT2, RT2>& rhs ) { return lhs->pParent == rhs.pParent && lhs->it == rhs.it; }

    template<typename ST, typename TT, typename TPT1, typename CT, typename PT1, typename RT1, typename TPT2, typename PT2, typename RT2>
    void sequential_it_init( sequential_iterator<ST, TT, TPT1, CT, PT1, RT1>* dest, const sequential_iterator<ST, TT, TPT2, CT, PT2, RT2>& src ) { dest->it = src.it; dest->pParent = src.pParent; }

    template<typename ST, typename TT, typename TPT1, typename CT, typename PT1, typename RT1, typename TPT2, typename PT2, typename RT2>
    bool sequential_it_eq( const sequential_iterator<ST, TT, TPT1, CT, PT1, RT1>* lhs, const sequential_iterator<ST, TT, TPT2, CT, PT2, RT2>& rhs ) { return lhs->pParent == rhs.pParent && lhs->it == rhs.it; }

    template<typename ST, typename TT, typename TPT1, typename CT, typename PT1, typename RT1, typename TPT2, typename PT2, typename RT2>
    bool sequential_it_less( const sequential_iterator<ST, TT, TPT1, CT, PT1, RT1>* lhs, const sequential_iterator<ST, TT, TPT2, CT, PT2, RT2>& rhs ) { return lhs->pParent == rhs.pParent && lhs->it < rhs.it; }
}

template<typename stored_type, typename tree_type, typename tree_pointer_type, typename container_type, typename pointer_type, typename reference_type>
class tcl::associative_iterator : public std::iterator<std::bidirectional_iterator_tag, stored_type, ptrdiff_t, pointer_type, reference_type>
{
    public:

        associative_iterator() : pParent( 0 ) {}

        associative_iterator( const associative_iterator<stored_type, tree_type, tree_type*, container_type, stored_type*, stored_type&>& src ) { associative_it_init( this, src ); }
        virtual ~associative_iterator() {}
    protected:

        explicit associative_iterator( const typename container_type::const_iterator& iter, const associative_tree<stored_type, tree_type, container_type>* pCalled_node ) : it( iter ), pParent( pCalled_node ) {}

    public:

        reference_type operator*() const { return const_cast<reference_type>( *( *it )->get() );}
        pointer_type operator->() const { return const_cast<pointer_type>( ( *it )->get() );}

        associative_iterator& operator ++() { ++it; return *this;}
        associative_iterator operator ++( int ) { associative_iterator old( *this ); ++*this; return old;}
        associative_iterator& operator --() { --it; return *this;}
        associative_iterator operator --( int ) { associative_iterator old( *this ); --*this; return old;}


        tree_pointer_type node() const { return const_cast<tree_pointer_type>( *it );}


        bool operator == ( const associative_iterator<stored_type, tree_type, const tree_type*, container_type, const stored_type*, const stored_type&>& rhs ) const { return associative_it_eq( this, rhs ); }
        bool operator != ( const associative_iterator<stored_type, tree_type, const tree_type*, container_type, const stored_type*, const stored_type&>& rhs ) const { return !( *this == rhs ); }

    protected:

        typename container_type::const_iterator it;
        const associative_tree<stored_type, tree_type, container_type>* pParent;
        // 123 "child_iterator.h"
        template<typename T, typename U, typename V> friend class basic_tree;
        template<typename T, typename U, typename V> friend class associative_tree;
        template<typename T, typename U, typename V, typename W, typename X, typename Y, typename Z> friend class pre_order_descendant_iterator;
        template<typename T, typename U, typename V, typename W, typename X, typename Y, typename Z> friend class post_order_descendant_iterator;
        friend void associative_it_init<>( associative_iterator<stored_type, tree_type, const tree_type*, container_type, const stored_type*, const stored_type&>*, const associative_iterator<stored_type, tree_type, tree_type*, container_type, stored_type*, stored_type&>& );
        friend void associative_it_init<>( associative_iterator<stored_type, tree_type, tree_type*, container_type, stored_type*, stored_type&>*, const associative_iterator<stored_type, tree_type, tree_type*, container_type, stored_type*, stored_type&>& );
        friend bool associative_it_eq<>( const associative_iterator<stored_type, tree_type, const tree_type*, container_type, const stored_type*, const stored_type&>*, const associative_iterator<stored_type, tree_type, const tree_type*, container_type, const stored_type*, const stored_type&>& );
        friend bool associative_it_eq<>( const associative_iterator<stored_type, tree_type, tree_type*, container_type, stored_type*, stored_type&>*, const associative_iterator<stored_type, tree_type, const tree_type*, container_type, const stored_type*, const stored_type&>& );

};

template<typename stored_type, typename tree_type, typename tree_pointer_type, typename container_type, typename pointer_type, typename reference_type>
class tcl::sequential_iterator : public std::iterator<std::random_access_iterator_tag, stored_type, ptrdiff_t, pointer_type, reference_type>
{
    public:

        sequential_iterator() : pParent( 0 ) {}

        sequential_iterator( const sequential_iterator<stored_type, tree_type, tree_type*, container_type, stored_type*, stored_type&>& src ) { sequential_it_init( this, src );}
        virtual ~sequential_iterator() {}
    protected:



        explicit sequential_iterator( typename container_type::const_iterator iter, const tree_type* pCalled_node ) : it( iter ), pParent( pCalled_node ) {}


    public:

        typedef size_t size_type;



        typedef typename std::iterator_traits<sequential_iterator>::difference_type difference_type;



        reference_type operator*() const { return const_cast<reference_type>( *( *it )->get() );}
        pointer_type operator->() const { return const_cast<pointer_type>( ( *it )->get() );}

        sequential_iterator& operator ++() { ++it; return *this;}
        sequential_iterator operator ++( int ) { sequential_iterator old( *this ); ++*this; return old;}
        sequential_iterator& operator --() { --it; return *this;}
        sequential_iterator operator --( int ) { sequential_iterator old( *this ); --*this; return old;}
        sequential_iterator& operator +=( difference_type n ) { it += n; return *this;}
        sequential_iterator& operator -=( difference_type n ) { it -= n; return *this;}
        difference_type operator -( const sequential_iterator& rhs ) const { return it - rhs.it;}
        tree_pointer_type node() const { return const_cast<tree_pointer_type>( *it );}

        bool operator == ( const sequential_iterator<stored_type, tree_type, const tree_type*, container_type, const stored_type*, const stored_type&>& rhs ) const { return sequential_it_eq( this, rhs ); }
        bool operator != ( const sequential_iterator<stored_type, tree_type, const tree_type*, container_type, const stored_type*, const stored_type&>& rhs ) const { return !( *this == rhs ); }
        bool operator < ( const sequential_iterator<stored_type, tree_type, const tree_type*, container_type, const stored_type*, const stored_type&>& rhs ) const { return sequential_it_less( this, rhs ); }
        bool operator <= ( const sequential_iterator<stored_type, tree_type, const tree_type*, container_type, const stored_type*, const stored_type&>& rhs ) const { return *this < rhs || *this == rhs; }
        bool operator > ( const sequential_iterator<stored_type, tree_type, const tree_type*, container_type, const stored_type*, const stored_type&>& rhs ) const { return !( *this <= rhs ); }
        bool operator >= ( const sequential_iterator<stored_type, tree_type, const tree_type*, container_type, const stored_type*, const stored_type&>& rhs ) const { return !( *this < rhs ); }

    protected:

        typename container_type::const_iterator it;
        const tree_type* pParent;
        template<typename T> friend class sequential_tree;
        template<typename T, typename U, typename V, typename W, typename X, typename Y, typename Z> friend class pre_order_descendant_iterator;
        template<typename T, typename U, typename V, typename W, typename X, typename Y, typename Z> friend class post_order_descendant_iterator;
        friend void sequential_it_init<>( sequential_iterator<stored_type, tree_type, const tree_type*, container_type, const stored_type*, const stored_type&>*, const sequential_iterator<stored_type, tree_type, tree_type*, container_type, stored_type*, stored_type&>& );
        friend void sequential_it_init<>( sequential_iterator<stored_type, tree_type, tree_type*, container_type, stored_type*, stored_type&>*, const sequential_iterator<stored_type, tree_type, tree_type*, container_type, stored_type*, stored_type&>& );
        friend bool sequential_it_eq<>( const sequential_iterator<stored_type, tree_type, const tree_type*, container_type, const stored_type*, const stored_type&>*, const sequential_iterator<stored_type, tree_type, const tree_type*, container_type, const stored_type*, const stored_type&>& );
        friend bool sequential_it_eq<>( const sequential_iterator<stored_type, tree_type, tree_type*, container_type, stored_type*, stored_type&>*, const sequential_iterator<stored_type, tree_type, const tree_type*, container_type, const stored_type*, const stored_type&>& );
        friend bool sequential_it_less<>( const sequential_iterator<stored_type, tree_type, const tree_type*, container_type, const stored_type*, const stored_type&>*, const sequential_iterator<stored_type, tree_type, const tree_type*, container_type, const stored_type*, const stored_type&>& );
        friend bool sequential_it_less<>( const sequential_iterator<stored_type, tree_type, tree_type*, container_type, stored_type*, stored_type&>*, const sequential_iterator<stored_type, tree_type, const tree_type*, container_type, const stored_type*, const stored_type&>& );

};

namespace tcl
{

    template<typename T, typename U, typename V> class basic_tree;
    template<typename T> class sequential_tree;
    template<typename T, typename U, typename V> class associative_tree;
    template<typename T, typename U, typename V, typename W, typename X, typename Z> class associative_reverse_iterator;
    template<typename T, typename U, typename V, typename W, typename X, typename Z> class sequential_reverse_iterator;
    template<typename T, typename U, typename V, typename W, typename X> class associative_reverse_node_iterator;
    template<typename T, typename U, typename V, typename W, typename X> class sequential_reverse_node_iterator;

    template<typename T, typename U, typename V, typename W, typename X> class associative_node_iterator;
    template<typename T, typename U, typename V, typename W, typename X> class sequential_node_iterator;
    template<typename T, typename U, typename V, typename W, typename X, typename Y> class pre_order_descendant_node_iterator;
    template<typename T, typename U, typename V, typename W, typename X, typename Y> class post_order_descendant_node_iterator;

    template<typename ST, typename TT, typename CT, typename PT1, typename RT1, typename PT2, typename RT2>
    void associative_node_it_init( associative_node_iterator<ST, TT, CT, PT1, RT1>* dest, const associative_node_iterator<ST, TT, CT, PT2, RT2>& src ) { dest->it = src.it; dest->pParent = src.pParent; }

    template<typename ST, typename TT, typename CT, typename PT1, typename RT1, typename PT2, typename RT2>
    bool associative_node_it_eq( const associative_node_iterator<ST, TT, CT, PT1, RT1>* lhs, const associative_node_iterator<ST, TT, CT, PT2, RT2>& rhs ) { return lhs->pParent == rhs.pParent && lhs->it == rhs.it; }

    template<typename ST, typename TT, typename CT, typename PT1, typename RT1, typename PT2, typename RT2>
    void sequential_node_it_init( sequential_node_iterator<ST, TT, CT, PT1, RT1>* dest, const sequential_node_iterator<ST, TT, CT, PT2, RT2>& src ) { dest->it = src.it; dest->pParent = src.pParent; }

    template<typename ST, typename TT, typename CT, typename PT1, typename RT1, typename PT2, typename RT2>
    bool sequential_node_it_eq( const sequential_node_iterator<ST, TT, CT, PT1, RT1>* lhs, const sequential_node_iterator<ST, TT, CT, PT2, RT2>& rhs ) { return lhs->pParent == rhs.pParent && lhs->it == rhs.it; }

    template<typename ST, typename TT, typename CT, typename PT1, typename RT1, typename PT2, typename RT2>
    bool sequential_node_it_less( const sequential_node_iterator<ST, TT, CT, PT1, RT1>* lhs, const sequential_node_iterator<ST, TT, CT, PT2, RT2>& rhs ) { return lhs->pParent == rhs.pParent && lhs->it < rhs.it; }
}

template<typename stored_type, typename tree_type, typename container_type, typename pointer_type, typename reference_type>
class tcl::associative_node_iterator : public std::iterator<std::bidirectional_iterator_tag, tree_type, ptrdiff_t, pointer_type, reference_type>
{
    public:

        associative_node_iterator() : pParent( 0 ) {}

        associative_node_iterator( const associative_node_iterator<stored_type, tree_type, container_type, tree_type*, tree_type&>& src ) { associative_node_it_init( this, src ); }

    protected:

        explicit associative_node_iterator( const typename container_type::const_iterator& iter, const associative_tree<stored_type, tree_type, container_type>* pCalled_node ) : it( iter ), pParent( pCalled_node ) {}

    public:

        reference_type operator*() const { return const_cast<reference_type>( *( *it ) );}
        pointer_type operator->() const { return const_cast<pointer_type>( *it );}

        associative_node_iterator& operator ++() { ++it; return *this;}
        associative_node_iterator operator ++( int ) { associative_node_iterator old( *this ); ++*this; return old;}
        associative_node_iterator& operator --() { --it; return *this;}
        associative_node_iterator operator --( int ) { associative_node_iterator old( *this ); --*this; return old;}


        bool operator == ( const associative_node_iterator<stored_type, tree_type, container_type, const tree_type*, const tree_type&>& rhs ) const { return associative_node_it_eq( this, rhs ); }
        bool operator != ( const associative_node_iterator<stored_type, tree_type, container_type, const tree_type*, const tree_type&>& rhs ) const { return !( *this == rhs ); }

    protected:

        typename container_type::const_iterator it;
        const associative_tree<stored_type, tree_type, container_type>* pParent;
        template<typename T, typename U, typename V> friend class basic_tree;
        template<typename T, typename U, typename V> friend class associative_tree;
        template<typename T, typename U, typename V, typename W, typename X, typename Y> friend class pre_order_descendant_node_iterator;
        template<typename T, typename U, typename V, typename W, typename X, typename Y> friend class post_order_descendant_node_iterator;
        friend void associative_node_it_init<>( associative_node_iterator<stored_type, tree_type, container_type, const tree_type*, const tree_type&>*, const associative_node_iterator<stored_type, tree_type, container_type, tree_type*, tree_type&>& );
        friend void associative_node_it_init<>( associative_node_iterator<stored_type, tree_type, container_type, tree_type*, tree_type&>*, const associative_node_iterator<stored_type, tree_type, container_type, tree_type*, tree_type&>& );
        friend bool associative_node_it_eq<>( const associative_node_iterator<stored_type, tree_type, container_type, const tree_type*, const tree_type&>*, const associative_node_iterator<stored_type, tree_type, container_type, const tree_type*, const tree_type&>& );
        friend bool associative_node_it_eq<>( const associative_node_iterator<stored_type, tree_type, container_type, tree_type*, tree_type&>*, const associative_node_iterator<stored_type, tree_type, container_type, const tree_type*, const tree_type&>& );

};

template<typename stored_type, typename tree_type, typename container_type, typename pointer_type, typename reference_type>
class tcl::sequential_node_iterator : public std::iterator<std::bidirectional_iterator_tag, tree_type, ptrdiff_t, pointer_type, reference_type>
{
    public:

        sequential_node_iterator() : pParent( 0 ) {}

        sequential_node_iterator( const sequential_node_iterator<stored_type, tree_type, container_type, tree_type*, tree_type&>& src ) { sequential_node_it_init( this, src ); }

    protected:

        explicit sequential_node_iterator( typename container_type::const_iterator it_, const tree_type* pParent_ ) : it( it_ ), pParent( pParent_ ) {}

    public:

        typedef size_t size_type;

        typedef typename std::iterator_traits<sequential_node_iterator>::difference_type difference_type;

        reference_type operator*() const { return const_cast<reference_type>( *( *it ) );}
        pointer_type operator->() const { return const_cast<pointer_type>( *it );}

        sequential_node_iterator& operator ++() { ++it; return *this;}
        sequential_node_iterator operator ++( int ) { sequential_node_iterator old( *this ); ++*this; return old;}
        sequential_node_iterator& operator --() { --it; return *this;}
        sequential_node_iterator operator --( int ) { sequential_node_iterator old( *this ); --*this; return old;}
        sequential_node_iterator& operator +=( size_type n ) { it += n; return *this;}
        sequential_node_iterator& operator -=( size_type n ) { it -= n; return *this;}
        difference_type operator -( const sequential_node_iterator& rhs ) const { return it - rhs.it;}

        bool operator == ( const sequential_node_iterator<stored_type, tree_type, container_type, const tree_type*, const tree_type&>& rhs ) const { return sequential_node_it_eq( this, rhs ); }
        bool operator != ( const sequential_node_iterator<stored_type, tree_type, container_type, const tree_type*, const tree_type&>& rhs ) const { return !( *this == rhs ); }
        bool operator < ( const sequential_node_iterator<stored_type, tree_type, container_type, const tree_type*, const tree_type&>& rhs ) const { return sequential_node_it_less( this, rhs ); }
        bool operator <= ( const sequential_node_iterator<stored_type, tree_type, container_type, const tree_type*, const tree_type&>& rhs ) const { return *this < rhs || *this == rhs; }
        bool operator > ( const sequential_node_iterator<stored_type, tree_type, container_type, const tree_type*, const tree_type&>& rhs ) const { return !( *this <= rhs ); }
        bool operator >= ( const sequential_node_iterator<stored_type, tree_type, container_type, const tree_type*, const tree_type&>& rhs ) const { return !( *this < rhs ); }

    protected:

        typename container_type::const_iterator it;
        const tree_type* pParent;
        // 202 "child_node_iterator.h"
        template<typename T> friend class sequential_tree;
        template<typename T, typename U, typename V, typename W, typename X, typename Y> friend class pre_order_descendant_node_iterator;
        template<typename T, typename U, typename V, typename W, typename X, typename Y> friend class post_order_descendant_node_iterator;
        friend void sequential_node_it_init<>( sequential_node_iterator<stored_type, tree_type, container_type, const tree_type*, const tree_type&>*, const sequential_node_iterator<stored_type, tree_type, container_type, tree_type*, tree_type&>& );
        friend void sequential_node_it_init<>( sequential_node_iterator<stored_type, tree_type, container_type, tree_type*, tree_type&>*, const sequential_node_iterator<stored_type, tree_type, container_type, tree_type*, tree_type&>& );
        friend bool sequential_node_it_eq<>( const sequential_node_iterator<stored_type, tree_type, container_type, const tree_type*, const tree_type&>*, const sequential_node_iterator<stored_type, tree_type, container_type, const tree_type*, const tree_type&>& );
        friend bool sequential_node_it_eq<>( const sequential_node_iterator<stored_type, tree_type, container_type, tree_type*, tree_type&>*, const sequential_node_iterator<stored_type, tree_type, container_type, const tree_type*, const tree_type&>& );
        friend bool sequential_node_it_less<>( const sequential_node_iterator<stored_type, tree_type, container_type, const tree_type*, const tree_type&>*, const sequential_node_iterator<stored_type, tree_type, container_type, const tree_type*, const tree_type&>& );
        friend bool sequential_node_it_less<>( const sequential_node_iterator<stored_type, tree_type, container_type, tree_type*, tree_type&>*, const sequential_node_iterator<stored_type, tree_type, container_type, const tree_type*, const tree_type&>& );

};

namespace tcl
{
    template<typename T, typename U, typename V, typename W, typename X, typename Y, typename Z> class pre_order_descendant_iterator;
    template<typename T, typename U, typename V, typename W, typename X, typename Y, typename Z> class post_order_descendant_iterator;
    template<typename T, typename U, typename V, typename W, typename X, typename Y, typename Z> class level_order_descendant_iterator;
    template<typename T, typename U> class tree;
    template<typename T, typename U> class multitree;
    template<typename T, typename U, typename V> class unique_tree;

    template<typename ST, typename TT, typename TPT1, typename CT, typename BIT1, typename PT1, typename RT1, typename TPT2, typename BIT2, typename PT2, typename RT2>
    void pre_order_it_init( pre_order_descendant_iterator<ST, TT, TPT1, CT, BIT1, PT1, RT1>* dest, const pre_order_descendant_iterator<ST, TT, TPT2, CT, BIT2, PT2, RT2>& src )
    {
        dest->it = src.it;
        dest->pTop_node = src.pTop_node;
        dest->at_top = src.at_top;
        std::stack<typename TT::iterator> s( src.node_stack );
        std::stack<typename TT::iterator> temp;

        while ( !s.empty() ) { temp.push( s.top() ); s.pop(); }

        while ( !temp.empty() ) { dest->node_stack.push( temp.top() ); temp.pop(); }
    }

    template<typename ST, typename TT, typename TPT1, typename CT, typename BIT1, typename PT1, typename RT1, typename TPT2, typename BIT2, typename PT2, typename RT2>
    bool pre_order_it_eq( const pre_order_descendant_iterator<ST, TT, TPT1, CT, BIT1, PT1, RT1>* lhs, const pre_order_descendant_iterator<ST, TT, TPT2, CT, BIT2, PT2, RT2>& rhs ) { return lhs->it == rhs.it && lhs->at_top == rhs.at_top; }

    template<typename ST, typename TT, typename TPT1, typename CT, typename BIT1, typename PT1, typename RT1, typename TPT2, typename BIT2, typename PT2, typename RT2>
    void post_order_it_init( post_order_descendant_iterator<ST, TT, TPT1, CT, BIT1, PT1, RT1>* dest, const post_order_descendant_iterator<ST, TT, TPT2, CT, BIT2, PT2, RT2>& src )
    {
        dest->it = src.it;
        dest->pTop_node = src.pTop_node;
        dest->at_top = src.at_top;
        std::stack<typename TT::iterator> s( src.node_stack );
        std::stack<typename TT::iterator> temp;

        while ( !s.empty() ) { temp.push( s.top() ); s.pop(); }

        while ( !temp.empty() ) { dest->node_stack.push( temp.top() ); temp.pop(); }
    }

    template<typename ST, typename TT, typename TPT1, typename CT, typename BIT1, typename PT1, typename RT1, typename TPT2, typename BIT2, typename PT2, typename RT2>
    bool post_order_it_eq( const post_order_descendant_iterator<ST, TT, TPT1, CT, BIT1, PT1, RT1>* lhs, const post_order_descendant_iterator<ST, TT, TPT2, CT, BIT2, PT2, RT2>& rhs ) { return lhs->it == rhs.it && lhs->at_top == rhs.at_top; }

    template<typename ST, typename TT, typename TPT1, typename CT, typename BIT1, typename PT1, typename RT1, typename TPT2, typename BIT2, typename PT2, typename RT2>
    void level_order_it_init( level_order_descendant_iterator<ST, TT, TPT1, CT, BIT1, PT1, RT1>* dest, const level_order_descendant_iterator<ST, TT, TPT2, CT, BIT2, PT2, RT2>& src )
    {
        dest->it = src.it;
        dest->pTop_node = src.pTop_node;
        dest->at_top = src.at_top;
        std::queue<typename TT::iterator> temp( src.node_queue );

        while ( !temp.empty() ) {dest->node_queue.push( temp.front() ); temp.pop();}
    }

    template<typename ST, typename TT, typename TPT1, typename CT, typename BIT1, typename PT1, typename RT1, typename TPT2, typename BIT2, typename PT2, typename RT2>
    bool level_order_it_eq( const level_order_descendant_iterator<ST, TT, TPT1, CT, BIT1, PT1, RT1>* lhs, const level_order_descendant_iterator<ST, TT, TPT2, CT, BIT2, PT2, RT2>& rhs ) { return lhs->it == rhs.it && lhs->at_top == rhs.at_top; }
}

template<typename stored_type, typename tree_type, typename tree_pointer_type, typename container_type, typename base_iterator_type, typename pointer_type, typename reference_type>
class tcl::pre_order_descendant_iterator
    : public std::iterator<std::bidirectional_iterator_tag, stored_type, ptrdiff_t, pointer_type, reference_type>
{
    public:

        pre_order_descendant_iterator() : pTop_node( 0 ), at_top( false ) {}

        pre_order_descendant_iterator( const pre_order_descendant_iterator<stored_type, tree_type, tree_type*, container_type, typename tree_type::iterator, stored_type*, stored_type&>& src ) { pre_order_it_init( this, src ); }
        virtual ~pre_order_descendant_iterator() {}

    private:
        explicit pre_order_descendant_iterator( tree_pointer_type pCalled_node, bool beg ) : it( beg ? pCalled_node->begin() : pCalled_node->end() ), pTop_node( pCalled_node ), at_top( beg ) {}

    public:

        pre_order_descendant_iterator& operator ++();
        pre_order_descendant_iterator operator ++( int ) { pre_order_descendant_iterator old( *this ); ++*this; return old;}
        pre_order_descendant_iterator& operator --();
        pre_order_descendant_iterator operator --( int ) { pre_order_descendant_iterator old( *this ); --*this; return old;}


        reference_type operator*() const { return at_top ? *pTop_node->get() : it.operator * ();}
        pointer_type operator->() const { return at_top ? pTop_node->get() : it.operator ->();}
        tree_pointer_type node() const { return at_top ? pTop_node : it.node();}
        base_iterator_type base() const { return it; }



        bool operator == ( const pre_order_descendant_iterator<stored_type, tree_type, const tree_type*, container_type, typename tree_type::const_iterator, const stored_type*, const stored_type&>& rhs ) const { return pre_order_it_eq( this, rhs ); }
        bool operator != ( const pre_order_descendant_iterator<stored_type, tree_type, const tree_type*, container_type, typename tree_type::const_iterator, const stored_type*, const stored_type&>& rhs ) const { return !( *this == rhs ); }

    private:

        base_iterator_type it;
        std::stack<base_iterator_type> node_stack;
        tree_pointer_type pTop_node;
        typename container_type::const_reverse_iterator rit;
        bool at_top;
        // 146 "descendant_iterator.h"
        template<typename T, typename U> friend class tree;
        template<typename T, typename U> friend class multitree;
        template<typename T, typename U, typename V> friend class unique_tree;
        template<typename T> friend class sequential_tree;
        friend void pre_order_it_init<>( pre_order_descendant_iterator<stored_type, tree_type, const tree_type*, container_type, typename tree_type::const_iterator, const stored_type*, const stored_type&>*, const pre_order_descendant_iterator<stored_type, tree_type, tree_type*, container_type, typename tree_type::iterator, stored_type*, stored_type&>& );
        friend void pre_order_it_init<>( pre_order_descendant_iterator<stored_type, tree_type, tree_type*, container_type, typename tree_type::iterator, stored_type*, stored_type&>*, const pre_order_descendant_iterator<stored_type, tree_type, tree_type*, container_type, typename tree_type::iterator, stored_type*, stored_type&>& );
        friend bool pre_order_it_eq<>( const pre_order_descendant_iterator<stored_type, tree_type, const tree_type*, container_type, typename tree_type::const_iterator, const stored_type*, const stored_type&>*, const pre_order_descendant_iterator<stored_type, tree_type, const tree_type*, container_type, typename tree_type::const_iterator, const stored_type*, const stored_type&>& );
        friend bool pre_order_it_eq<>( const pre_order_descendant_iterator<stored_type, tree_type, tree_type*, container_type, typename tree_type::iterator, stored_type*, stored_type&>*, const pre_order_descendant_iterator<stored_type, tree_type, const tree_type*, container_type, typename tree_type::const_iterator, const stored_type*, const stored_type&>& );

};

template<typename stored_type, typename tree_type, typename tree_pointer_type, typename container_type, typename base_iterator_type, typename pointer_type, typename reference_type>
class tcl::post_order_descendant_iterator : public std::iterator<std::bidirectional_iterator_tag, stored_type, ptrdiff_t, pointer_type, reference_type>
{
    public:

        post_order_descendant_iterator() : pTop_node( 0 ) {}
        virtual ~post_order_descendant_iterator() {}

        post_order_descendant_iterator( const post_order_descendant_iterator<stored_type, tree_type, tree_type*, container_type, typename tree_type::iterator, stored_type*, stored_type&>& src ) { post_order_it_init( this, src ); }

    private:
        explicit post_order_descendant_iterator( tree_pointer_type pCalled_node, bool beg );

    public:

        post_order_descendant_iterator& operator ++();
        post_order_descendant_iterator operator ++( int ) { post_order_descendant_iterator old( *this ); ++*this; return old;}
        post_order_descendant_iterator& operator --();
        post_order_descendant_iterator operator --( int ) { post_order_descendant_iterator old( *this ); --*this; return old;}


        reference_type operator*() const { return at_top ? *pTop_node->get() : it.operator * ();}
        pointer_type operator->() const { return at_top ? pTop_node->get() : it.operator ->();}
        tree_pointer_type node() const { return at_top ? pTop_node : it.node();}
        base_iterator_type base() const { return it; }


        bool operator == ( const post_order_descendant_iterator<stored_type, tree_type, const tree_type*, container_type, typename tree_type::const_iterator, const stored_type*, const stored_type&>& rhs ) const { return post_order_it_eq( this, rhs ); }
        bool operator != ( const post_order_descendant_iterator<stored_type, tree_type, const tree_type*, container_type, typename tree_type::const_iterator, const stored_type*, const stored_type&>& rhs ) const { return !( *this == rhs ); }

    private:

        std::stack<base_iterator_type> node_stack;
        base_iterator_type it;
        tree_pointer_type pTop_node;
        typename container_type::const_reverse_iterator rit;
        bool at_top;
        // 215 "descendant_iterator.h"
        template<typename T> friend class sequential_tree;
        template<typename T, typename U> friend class tree;
        template<typename T, typename U> friend class multitree;
        template<typename T, typename U, typename V> friend class unique_tree;
        friend void post_order_it_init<>( post_order_descendant_iterator<stored_type, tree_type, const tree_type*, container_type, typename tree_type::const_iterator, const stored_type*, const stored_type&>*, const post_order_descendant_iterator<stored_type, tree_type, tree_type*, container_type, typename tree_type::iterator, stored_type*, stored_type&>& );
        friend void post_order_it_init<>( post_order_descendant_iterator<stored_type, tree_type, tree_type*, container_type, typename tree_type::iterator, stored_type*, stored_type&>*, const post_order_descendant_iterator<stored_type, tree_type, tree_type*, container_type, typename tree_type::iterator, stored_type*, stored_type&>& );
        friend bool post_order_it_eq<>( const post_order_descendant_iterator<stored_type, tree_type, const tree_type*, container_type, typename tree_type::const_iterator, const stored_type*, const stored_type&>*, const post_order_descendant_iterator<stored_type, tree_type, const tree_type*, container_type, typename tree_type::const_iterator, const stored_type*, const stored_type&>& );
        friend bool post_order_it_eq<>( const post_order_descendant_iterator<stored_type, tree_type, tree_type*, container_type, typename tree_type::iterator, stored_type*, stored_type&>*, const post_order_descendant_iterator<stored_type, tree_type, const tree_type*, container_type, typename tree_type::const_iterator, const stored_type*, const stored_type&>& );
};

template<typename stored_type, typename tree_type, typename tree_pointer_type, typename container_type, typename base_iterator_type, typename pointer_type, typename reference_type>
class tcl::level_order_descendant_iterator : public std::iterator<std::forward_iterator_tag, stored_type, ptrdiff_t, pointer_type, reference_type>
{
    public:

        level_order_descendant_iterator() : pTop_node( 0 ), at_top( false ) {}

        level_order_descendant_iterator( const level_order_descendant_iterator<stored_type, tree_type, tree_type*, container_type, typename tree_type::iterator, stored_type*, stored_type&>& src ) { level_order_it_init( this, src ); }
        virtual ~level_order_descendant_iterator() {}

    protected:
        explicit level_order_descendant_iterator( tree_pointer_type pCalled_node, bool beg ) : it( beg ? pCalled_node->begin() : pCalled_node->end() ), pTop_node( pCalled_node ), at_top( beg ) {}

    public:

        level_order_descendant_iterator& operator ++();
        level_order_descendant_iterator operator ++( int ) { level_order_descendant_iterator old( *this ); ++*this; return old;}


        reference_type operator*() const { return at_top ? *pTop_node->get() : it.operator * ();}
        pointer_type operator->() const { return at_top ? pTop_node->get() : it.operator ->();}
        tree_pointer_type node() const { return at_top ? pTop_node : it.node();}
        base_iterator_type base() const { return it; }


        bool operator == ( const level_order_descendant_iterator<stored_type, tree_type, const tree_type*, container_type, typename tree_type::const_iterator, const stored_type*, const stored_type&>& rhs ) const { return level_order_it_eq( this, rhs ); }
        bool operator != ( const level_order_descendant_iterator<stored_type, tree_type, const tree_type*, container_type, typename tree_type::const_iterator, const stored_type*, const stored_type&>& rhs ) const { return !( *this == rhs ); }

    private:

        base_iterator_type it;
        std::queue<base_iterator_type> node_queue;
        tree_pointer_type pTop_node;
        bool at_top;
        // 281 "descendant_iterator.h"
        template<typename T, typename U> friend class tree;
        template<typename T, typename U> friend class multitree;
        template<typename T, typename U, typename V> friend class unique_tree;
        template<typename T> friend class sequential_tree;
        friend void level_order_it_init<>( level_order_descendant_iterator<stored_type, tree_type, const tree_type*, container_type, typename tree_type::const_iterator, const stored_type*, const stored_type&>*, const level_order_descendant_iterator<stored_type, tree_type, tree_type*, container_type, typename tree_type::iterator, stored_type*, stored_type&>& );
        friend void level_order_it_init<>( level_order_descendant_iterator<stored_type, tree_type, tree_type*, container_type, typename tree_type::iterator, stored_type*, stored_type&>*, const level_order_descendant_iterator<stored_type, tree_type, tree_type*, container_type, typename tree_type::iterator, stored_type*, stored_type&>& );
        friend bool level_order_it_eq<>( const level_order_descendant_iterator<stored_type, tree_type, const tree_type*, container_type, typename tree_type::const_iterator, const stored_type*, const stored_type&>*, const level_order_descendant_iterator<stored_type, tree_type, const tree_type*, container_type, typename tree_type::const_iterator, const stored_type*, const stored_type&>& );
        friend bool level_order_it_eq<>( const level_order_descendant_iterator<stored_type, tree_type, tree_type*, container_type, typename tree_type::iterator, stored_type*, stored_type&>*, const level_order_descendant_iterator<stored_type, tree_type, const tree_type*, container_type, typename tree_type::const_iterator, const stored_type*, const stored_type&>& );

};

template<typename stored_type, typename tree_type, typename tree_pointer_type, typename container_type, typename base_iterator_type, typename pointer_type, typename reference_type>
tcl::pre_order_descendant_iterator<stored_type, tree_type, tree_pointer_type, container_type, base_iterator_type, pointer_type, reference_type>& tcl::pre_order_descendant_iterator<stored_type, tree_type, tree_pointer_type, container_type, base_iterator_type, pointer_type, reference_type>::operator ++()
{
    if ( at_top )
    {
        at_top = false;
        it = pTop_node->begin();
    }

    else if ( !it.node()->empty() )
    {
        node_stack.push( it );
        it = it.node()->begin();
    }

    else
    {
        ++it;

        while ( !node_stack.empty() && it == ( node_stack.top() ).node()->end() )
        {
            it = node_stack.top();
            node_stack.pop();
            ++it;
        }
    }

    return *this;
}


template<typename stored_type, typename tree_type, typename tree_pointer_type, typename container_type, typename base_iterator_type, typename pointer_type, typename reference_type>
tcl::pre_order_descendant_iterator<stored_type, tree_type, tree_pointer_type, container_type, base_iterator_type, pointer_type, reference_type>& tcl::pre_order_descendant_iterator<stored_type, tree_type, tree_pointer_type, container_type, base_iterator_type, pointer_type, reference_type>::operator --()
{
    if ( it == pTop_node->end() )
    {
        if ( pTop_node->empty() )
        {
            at_top = true;
            return *this;
        }

        rit = pTop_node->children.rbegin();

        if ( rit != const_cast<const tree_type*>( pTop_node )->children.rend() )
        {
            if ( !( *rit )->empty() )
            {
                do
                {
                    ++rit;
                    it = base_iterator_type( rit.base(), ( it != pTop_node->end() ? it.node() : pTop_node ) );
                    node_stack.push( it );
                    rit = it.node()->children.rbegin();
                }
                while ( !( *rit )->empty() );
            }

            ++rit;
            it = base_iterator_type( rit.base(), ( it != pTop_node->end() ? node() : pTop_node ) );
        }
    }

    else
    {
        if ( it != it.node()->parent()->begin() )
        {
            --it;

            if ( !it.node()->empty() )
            {
                do
                {
                    node_stack.push( it );
                    it = base_iterator_type( it.node()->children.end(), it.node() );
                    --it;
                }
                while ( !it.node()->empty() );
            }
        }

        else if ( !node_stack.empty() )
        {
            it = node_stack.top();
            node_stack.pop();
        }

        else
        {
            if ( !at_top )
            {
                at_top = true;
            }

            else
            {
                --it;
                at_top = false;
            }
        }
    }

    return *this;
}


template<typename stored_type, typename tree_type, typename tree_pointer_type, typename container_type, typename base_iterator_type, typename pointer_type, typename reference_type>
tcl::post_order_descendant_iterator<stored_type, tree_type, tree_pointer_type, container_type, base_iterator_type, pointer_type, reference_type>::post_order_descendant_iterator( tree_pointer_type pCalled_node, bool beg ) : pTop_node( pCalled_node ), at_top( false )
{
    if ( !beg )
    {
        it = pTop_node->end();
    }

    else
    {
        it = pCalled_node->begin();

        if ( it != pTop_node->end() )
        {
            if ( !it.node()->empty() )
            {
                do
                {
                    node_stack.push( it );
                    it = it.node()->begin();
                }
                while ( !it.node()->empty() );
            }
        }

        else
        {
            at_top = true;
        }
    }
}


template<typename stored_type, typename tree_type, typename tree_pointer_type, typename container_type, typename base_iterator_type, typename pointer_type, typename reference_type>
tcl::post_order_descendant_iterator<stored_type, tree_type, tree_pointer_type, container_type, base_iterator_type, pointer_type, reference_type>& tcl::post_order_descendant_iterator<stored_type, tree_type, tree_pointer_type, container_type, base_iterator_type, pointer_type, reference_type>::operator ++()
{
    if ( at_top )
    {
        at_top = false;
        it = pTop_node->end();
        return *this;
    }

    else if ( pTop_node->empty() )
    {
        ++it;
        return *this;
    }

    const base_iterator_type it_end = it.node()->parent()->end();
    ++it;

    if ( it != it_end && !it.node()->empty() )
    {
        do
        {
            node_stack.push( it );
            it = it.node()->begin();
        }
        while ( !it.node()->empty() );
    }

    else
    {
        if ( !node_stack.empty() && it == ( node_stack.top() ).node()->end() )
        {
            it = node_stack.top();
            node_stack.pop();
        }

        else if ( node_stack.empty() && it == pTop_node->end() )
        {
            at_top = true;
        }
    }

    return *this;
}


template<typename stored_type, typename tree_type, typename tree_pointer_type, typename container_type, typename base_iterator_type, typename pointer_type, typename reference_type>
tcl::post_order_descendant_iterator<stored_type, tree_type, tree_pointer_type, container_type, base_iterator_type, pointer_type, reference_type>& tcl::post_order_descendant_iterator<stored_type, tree_type, tree_pointer_type, container_type, base_iterator_type, pointer_type, reference_type>::operator --()
{
    if ( at_top )
    {
        at_top = false;
        typename container_type::const_reverse_iterator rit = pTop_node->children.rbegin();
        ++rit;
        it = base_iterator_type( rit.base(), pTop_node );
    }

    else if ( it == pTop_node->end() )
    {
        at_top = true;
    }

    else
    {
        if ( !it.node()->empty() )
        {
            typename container_type::const_reverse_iterator rit = it.node()->children.rbegin();
            node_stack.push( it );
            ++rit;
            it = base_iterator_type( rit.base(), it.node() );
        }

        else
        {
            if ( it != it.node()->parent()->begin() )
            {
                --it;
            }

            else
            {
                while ( !node_stack.empty() && it == node_stack.top().node()->begin() )
                {
                    it = node_stack.top();
                    node_stack.pop();
                }

                --it;
            }
        }
    }

    return *this;
}


template<typename stored_type, typename tree_type, typename tree_pointer_type, typename container_type, typename base_iterator_type, typename pointer_type, typename reference_type>
tcl::level_order_descendant_iterator<stored_type, tree_type, tree_pointer_type, container_type, base_iterator_type, pointer_type, reference_type>& tcl::level_order_descendant_iterator<stored_type, tree_type, tree_pointer_type, container_type, base_iterator_type, pointer_type, reference_type>::operator ++()
{
    if ( at_top )
    {
        at_top = false;
        it = pTop_node->begin();
        return *this;
    }

    const base_iterator_type it_end = it.node()->parent()->end();
    node_queue.push( it );
    ++it;

    if ( it == it_end )
    {
        while ( !node_queue.empty() )
        {
            it = node_queue.front();
            node_queue.pop();

            if ( !it.node()->empty() )
            {
                it = node()->begin();
                break;
            }

            else if ( node_queue.empty() )
            {
                it = pTop_node->end();
                return *this;
            }
        }
    }

    return *this;
}
// 294 "descendant_iterator.h" 2
// 32 "associative_tree.h" 2
// 1 "descendant_node_iterator.h" 1
// 27 "descendant_node_iterator.h"







namespace tcl
{
    template<typename T, typename U, typename V, typename W, typename X, typename Y> class pre_order_descendant_node_iterator;
    template<typename T, typename U, typename V, typename W, typename X, typename Y> class post_order_descendant_node_iterator;
    template<typename T, typename U, typename V, typename W, typename X, typename Y> class level_order_descendant_node_iterator;

    template<typename ST, typename TT, typename CT, typename BIT1, typename PT1, typename RT1, typename BIT2, typename PT2, typename RT2>
    void pre_order_node_it_init( pre_order_descendant_node_iterator<ST, TT, CT, BIT1, PT1, RT1>* dest, const pre_order_descendant_node_iterator<ST, TT, CT, BIT2, PT2, RT2>& src )
    {
        dest->it = src.it;
        dest->pTop_node = src.pTop_node;
        dest->at_top = src.at_top;
        std::stack<typename TT::node_iterator> s( src.node_stack );
        std::stack<typename TT::node_iterator> temp;

        while ( !s.empty() ) { temp.push( s.top() ); s.pop(); }

        while ( !temp.empty() ) { dest->node_stack.push( temp.top() ); temp.pop(); }
    }

    template<typename ST, typename TT, typename CT, typename BIT1, typename PT1, typename RT1, typename BIT2, typename PT2, typename RT2>
    bool pre_order_node_it_eq( const pre_order_descendant_node_iterator<ST, TT, CT, BIT1, PT1, RT1>* lhs, const pre_order_descendant_node_iterator<ST, TT, CT, BIT2, PT2, RT2>& rhs ) { return lhs->it == rhs.it && lhs->at_top == rhs.at_top; }

    template<typename ST, typename TT, typename CT, typename BIT1, typename PT1, typename RT1, typename BIT2, typename PT2, typename RT2>
    void post_order_node_it_init( post_order_descendant_node_iterator<ST, TT, CT, BIT1, PT1, RT1>* dest, const post_order_descendant_node_iterator<ST, TT, CT, BIT2, PT2, RT2>& src )
    {
        dest->it = src.it;
        dest->pTop_node = src.pTop_node;
        dest->at_top = src.at_top;
        std::stack<typename TT::node_iterator> s( src.node_stack );
        std::stack<typename TT::node_iterator> temp;

        while ( !s.empty() ) { temp.push( s.top() ); s.pop(); }

        while ( !temp.empty() ) { dest->node_stack.push( temp.top() ); temp.pop(); }
    }

    template<typename ST, typename TT, typename CT, typename BIT1, typename PT1, typename RT1, typename BIT2, typename PT2, typename RT2>
    bool post_order_node_it_eq( const post_order_descendant_node_iterator<ST, TT, CT, BIT1, PT1, RT1>* lhs, const post_order_descendant_node_iterator<ST, TT, CT, BIT2, PT2, RT2>& rhs ) { return lhs->it == rhs.it && lhs->at_top == rhs.at_top; }

    template<typename ST, typename TT, typename CT, typename BIT1, typename PT1, typename RT1, typename BIT2, typename PT2, typename RT2>
    void level_order_node_it_init( level_order_descendant_node_iterator<ST, TT, CT, BIT1, PT1, RT1>* dest, const level_order_descendant_node_iterator<ST, TT, CT, BIT2, PT2, RT2>& src )
    {
        dest->it = src.it;
        dest->pTop_node = src.pTop_node;
        dest->at_top = src.at_top;
        std::queue<typename TT::node_iterator> temp( src.node_queue );

        while ( !temp.empty() ) {dest->node_queue.push( temp.front() ); temp.pop();}
    }

    template<typename ST, typename TT, typename CT, typename BIT1, typename PT1, typename RT1, typename BIT2, typename PT2, typename RT2>
    bool level_order_node_it_eq( const level_order_descendant_node_iterator<ST, TT, CT, BIT1, PT1, RT1>* lhs, const level_order_descendant_node_iterator<ST, TT, CT, BIT2, PT2, RT2>& rhs ) { return lhs->it == rhs.it && lhs->at_top == rhs.at_top; }

}





template<typename stored_type, typename tree_type, typename container_type, typename base_iterator_type, typename pointer_type, typename reference_type>
class tcl::pre_order_descendant_node_iterator



    : public std::iterator<std::bidirectional_iterator_tag, tree_type, ptrdiff_t, pointer_type, reference_type>

{
    public:

        pre_order_descendant_node_iterator() : pTop_node( 0 ), at_top( false ) {}

        pre_order_descendant_node_iterator( const pre_order_descendant_node_iterator<stored_type, tree_type, container_type, typename tree_type::node_iterator, tree_type*, tree_type&>& src ) { pre_order_node_it_init( this, src ); }

    protected:
        explicit pre_order_descendant_node_iterator( pointer_type pCalled_node, bool beg ) : it( beg ? pCalled_node->node_begin() : pCalled_node->node_end() ), pTop_node( pCalled_node ), at_top( beg ) {}

    public:

        pre_order_descendant_node_iterator& operator ++();
        pre_order_descendant_node_iterator operator ++( int ) { pre_order_descendant_node_iterator old( *this ); ++*this; return old;}
        pre_order_descendant_node_iterator& operator --();
        pre_order_descendant_node_iterator operator --( int ) { pre_order_descendant_node_iterator old( *this ); --*this; return old;}

        reference_type operator*() const { return at_top ? *pTop_node : it.operator * ();}
        pointer_type operator->() const { return at_top ? pTop_node : it.operator ->();}
        base_iterator_type base() const { return it; }


        bool operator == ( const pre_order_descendant_node_iterator<stored_type, tree_type, container_type, typename tree_type::const_node_iterator, const tree_type*, const tree_type&>& rhs ) const { return pre_order_node_it_eq( this, rhs ); }
        bool operator != ( const pre_order_descendant_node_iterator<stored_type, tree_type, container_type, typename tree_type::const_node_iterator, const tree_type*, const tree_type&>& rhs ) const { return !( *this == rhs ); }

    private:

        base_iterator_type it;
        std::stack<base_iterator_type> node_stack;
        pointer_type pTop_node;
        typename container_type::const_reverse_iterator rit;
        bool at_top;
        // 139 "descendant_node_iterator.h"
        template<typename T, typename U> friend class tree;
        template<typename T, typename U> friend class multitree;
        template<typename T, typename U, typename V> friend class unique_tree;
        template<typename T> friend class sequential_tree;
        friend void pre_order_node_it_init<>( pre_order_descendant_node_iterator<stored_type, tree_type, container_type, typename tree_type::const_node_iterator, const tree_type*, const tree_type&>*, const pre_order_descendant_node_iterator<stored_type, tree_type, container_type, typename tree_type::node_iterator, tree_type*, tree_type&>& );
        friend void pre_order_node_it_init<>( pre_order_descendant_node_iterator<stored_type, tree_type, container_type, typename tree_type::node_iterator, tree_type*, tree_type&>*, const pre_order_descendant_node_iterator<stored_type, tree_type, container_type, typename tree_type::node_iterator, tree_type*, tree_type&>& );
        friend bool pre_order_node_it_eq<>( const pre_order_descendant_node_iterator<stored_type, tree_type, container_type, typename tree_type::const_node_iterator, const tree_type*, const tree_type&>*, const pre_order_descendant_node_iterator<stored_type, tree_type, container_type, typename tree_type::const_node_iterator, const tree_type*, const tree_type&>& );
        friend bool pre_order_node_it_eq<>( const pre_order_descendant_node_iterator<stored_type, tree_type, container_type, typename tree_type::node_iterator, tree_type*, tree_type&>*, const pre_order_descendant_node_iterator<stored_type, tree_type, container_type, typename tree_type::const_node_iterator, const tree_type*, const tree_type&>& );

};






template<typename stored_type, typename tree_type, typename container_type, typename base_iterator_type, typename pointer_type, typename reference_type>
class tcl::post_order_descendant_node_iterator



    : public std::iterator<std::bidirectional_iterator_tag, tree_type, ptrdiff_t, pointer_type, reference_type>

{
    public:

        post_order_descendant_node_iterator() : pTop_node( 0 ), at_top( false ) {}

        post_order_descendant_node_iterator( const post_order_descendant_node_iterator<stored_type, tree_type, container_type, typename tree_type::node_iterator, tree_type*, tree_type&>& src ) { post_order_node_it_init( this, src ); }

    private:
        explicit post_order_descendant_node_iterator( pointer_type pCalled_node, bool beg );

    public:

        post_order_descendant_node_iterator& operator ++();
        post_order_descendant_node_iterator operator ++( int ) { post_order_descendant_node_iterator old( *this ); ++*this; return old;}
        post_order_descendant_node_iterator& operator --();
        post_order_descendant_node_iterator operator --( int ) { post_order_descendant_node_iterator old( *this ); --*this; return old;}


        reference_type operator*() const { return at_top ? *pTop_node : it.operator * ();}
        pointer_type operator->() const { return at_top ? pTop_node : it.operator ->();}
        base_iterator_type base() const { return it; }


        bool operator == ( const post_order_descendant_node_iterator<stored_type, tree_type, container_type, typename tree_type::const_node_iterator, const tree_type*, const tree_type&>& rhs ) const { return post_order_node_it_eq( this, rhs ); }
        bool operator != ( const post_order_descendant_node_iterator<stored_type, tree_type, container_type, typename tree_type::const_node_iterator, const tree_type*, const tree_type&>& rhs ) const { return !( *this == rhs ); }

    private:

        std::stack<base_iterator_type> node_stack;
        base_iterator_type it;
        pointer_type pTop_node;
        typename container_type::const_reverse_iterator rit;
        bool at_top;
        // 206 "descendant_node_iterator.h"
        template<typename T, typename U> friend class tree;
        template<typename T, typename U> friend class multitree;
        template<typename T, typename U, typename V> friend class unique_tree;
        template<typename T> friend class sequential_tree;
        friend void post_order_node_it_init<>( post_order_descendant_node_iterator<stored_type, tree_type, container_type, typename tree_type::const_node_iterator, const tree_type*, const tree_type&>*, const post_order_descendant_node_iterator<stored_type, tree_type, container_type, typename tree_type::node_iterator, tree_type*, tree_type&>& );
        friend void post_order_node_it_init<>( post_order_descendant_node_iterator<stored_type, tree_type, container_type, typename tree_type::node_iterator, tree_type*, tree_type&>*, const post_order_descendant_node_iterator<stored_type, tree_type, container_type, typename tree_type::node_iterator, tree_type*, tree_type&>& );
        friend bool post_order_node_it_eq<>( const post_order_descendant_node_iterator<stored_type, tree_type, container_type, typename tree_type::const_node_iterator, const tree_type*, const tree_type&>*, const post_order_descendant_node_iterator<stored_type, tree_type, container_type, typename tree_type::const_node_iterator, const tree_type*, const tree_type&>& );
        friend bool post_order_node_it_eq<>( const post_order_descendant_node_iterator<stored_type, tree_type, container_type, typename tree_type::node_iterator, tree_type*, tree_type&>*, const post_order_descendant_node_iterator<stored_type, tree_type, container_type, typename tree_type::const_node_iterator, const tree_type*, const tree_type&>& );

};







template<typename stored_type, typename tree_type, typename container_type, typename base_iterator_type, typename pointer_type, typename reference_type>
class tcl::level_order_descendant_node_iterator



    : public std::iterator<std::forward_iterator_tag, tree_type, ptrdiff_t, pointer_type, reference_type>

{
    public:

        level_order_descendant_node_iterator() : pTop_node( 0 ), at_top( false ) {}

        level_order_descendant_node_iterator( const level_order_descendant_node_iterator<stored_type, tree_type, container_type, typename tree_type::node_iterator, tree_type*, tree_type&>& src ) { level_order_node_it_init( this, src ); }

    private:
        explicit level_order_descendant_node_iterator( pointer_type pCalled_node, bool beg ) : it( beg ? pCalled_node->node_begin() : pCalled_node->node_end() ), pTop_node( pCalled_node ), at_top( beg ) {}

    public:

        level_order_descendant_node_iterator& operator ++();
        level_order_descendant_node_iterator operator ++( int ) { level_order_descendant_node_iterator old( *this ); ++*this; return old;}


        reference_type operator*() const { return at_top ? const_cast<reference_type>( *pTop_node ) : const_cast<reference_type>( it.operator * () );}
        pointer_type operator->() const { return at_top ? const_cast<pointer_type>( pTop_node ) : const_cast<pointer_type>( it.operator ->() );}
        base_iterator_type base() const { return it; }


        bool operator == ( const level_order_descendant_node_iterator<stored_type, tree_type, container_type, typename tree_type::const_node_iterator, const tree_type*, const tree_type&>& rhs ) const { return level_order_node_it_eq( this, rhs ); }
        bool operator != ( const level_order_descendant_node_iterator<stored_type, tree_type, container_type, typename tree_type::const_node_iterator, const tree_type*, const tree_type&>& rhs ) const { return !( *this == rhs ); }

    private:

        base_iterator_type it;
        std::queue<base_iterator_type> node_queue;
        pointer_type pTop_node;
        bool at_top;
        // 271 "descendant_node_iterator.h"
        template<typename T, typename U> friend class tree;
        template<typename T, typename U> friend class multitree;
        template<typename T, typename U, typename V> friend class unique_tree;
        template<typename T> friend class sequential_tree;
        friend void level_order_node_it_init<>( level_order_descendant_node_iterator<stored_type, tree_type, container_type, typename tree_type::const_node_iterator, const tree_type*, const tree_type&>*, const level_order_descendant_node_iterator<stored_type, tree_type, container_type, typename tree_type::node_iterator, tree_type*, tree_type&>& );
        friend void level_order_node_it_init<>( level_order_descendant_node_iterator<stored_type, tree_type, container_type, typename tree_type::node_iterator, tree_type*, tree_type&>*, const level_order_descendant_node_iterator<stored_type, tree_type, container_type, typename tree_type::node_iterator, tree_type*, tree_type&>& );
        friend bool level_order_node_it_eq<>( const level_order_descendant_node_iterator<stored_type, tree_type, container_type, typename tree_type::const_node_iterator, const tree_type*, const tree_type&>*, const level_order_descendant_node_iterator<stored_type, tree_type, container_type, typename tree_type::const_node_iterator, const tree_type*, const tree_type&>& );
        friend bool level_order_node_it_eq<>( const level_order_descendant_node_iterator<stored_type, tree_type, container_type, typename tree_type::node_iterator, tree_type*, tree_type&>*, const level_order_descendant_node_iterator<stored_type, tree_type, container_type, typename tree_type::const_node_iterator, const tree_type*, const tree_type&>& );

};



// 1 "descendant_node_iterator.inl" 1
// 29 "descendant_node_iterator.inl"
template<typename stored_type, typename tree_type, typename container_type, typename base_iterator_type, typename pointer_type, typename reference_type>
tcl::pre_order_descendant_node_iterator<stored_type, tree_type, container_type, base_iterator_type, pointer_type, reference_type>& tcl::pre_order_descendant_node_iterator<stored_type, tree_type, container_type, base_iterator_type, pointer_type, reference_type>::operator ++()
{
    if ( at_top )
    {
        at_top = false;
        it = pTop_node->node_begin();
    }

    else if ( !it->empty() )
    {
        node_stack.push( it );
        it = it->node_begin();
    }

    else
    {
        ++it;

        while ( !node_stack.empty() && it == ( node_stack.top() )->node_end() )
        {
            it = node_stack.top();
            node_stack.pop();
            ++it;
        }
    }

    return *this;
}


template<typename stored_type, typename tree_type, typename container_type, typename base_iterator_type, typename pointer_type, typename reference_type>
tcl::pre_order_descendant_node_iterator<stored_type, tree_type, container_type, base_iterator_type, pointer_type, reference_type>& tcl::pre_order_descendant_node_iterator<stored_type, tree_type, container_type, base_iterator_type, pointer_type, reference_type>::operator --()
{
    if ( it == pTop_node->node_end() )
    {
        if ( pTop_node->empty() )
        {
            at_top = true;
            return *this;
        }

        rit = pTop_node->children.rbegin();

        if ( rit != const_cast<const tree_type*>( pTop_node )->children.rend() )
        {
            if ( !( *rit )->empty() )
            {
                do
                {
                    ++rit;
                    it = base_iterator_type( rit.base(), ( it != pTop_node->node_end() ? & ( *it ) : pTop_node ) );
                    node_stack.push( it );
                    rit = it->children.rbegin();
                }
                while ( !( *rit )->empty() );
            }

            ++rit;
            it = base_iterator_type( rit.base(), ( it != pTop_node->node_end() ? & ( *it ) : pTop_node ) );
        }
    }

    else
    {
        if ( it != it->parent()->node_begin() )
        {
            --it;

            if ( !it->empty() )
            {
                do
                {
                    node_stack.push( it );
                    it = base_iterator_type( it->children.end(), &( *it ) );
                    --it;
                }
                while ( !it->empty() );
            }
        }

        else if ( !node_stack.empty() )
        {
            it = node_stack.top();
            node_stack.pop();
        }

        else
        {
            if ( !at_top )
            {
                at_top = true;
            }

            else
            {
                --it;
                at_top = false;
            }
        }
    }

    return *this;
}


template<typename stored_type, typename tree_type, typename container_type, typename base_iterator_type, typename pointer_type, typename reference_type>
tcl::post_order_descendant_node_iterator<stored_type, tree_type, container_type, base_iterator_type, pointer_type, reference_type>::post_order_descendant_node_iterator( pointer_type pCalled_node, bool beg ) : pTop_node( pCalled_node ), at_top( false )
{
    if ( !beg )
    {
        it = pTop_node->node_end();
    }

    else
    {
        it = pTop_node->node_begin();

        if ( it != pTop_node->node_end() )
        {
            if ( !it->empty() )
            {
                do
                {
                    node_stack.push( it );
                    it = it->node_begin();
                }
                while ( !it->empty() );
            }
        }

        else
        {
            at_top = true;
        }
    }
}


template<typename stored_type, typename tree_type, typename container_type, typename base_iterator_type, typename pointer_type, typename reference_type>
tcl::post_order_descendant_node_iterator<stored_type, tree_type, container_type, base_iterator_type, pointer_type, reference_type>& tcl::post_order_descendant_node_iterator<stored_type, tree_type, container_type, base_iterator_type, pointer_type, reference_type>::operator ++()
{
    if ( at_top )
    {
        at_top = false;
        it = pTop_node->node_end();
        return *this;
    }

    else if ( pTop_node->empty() )
    {
        ++it;
        return *this;
    }

    const base_iterator_type it_end = it->parent()->node_end();
    ++it;

    if ( it != it_end && !it->empty() )
    {
        do
        {
            node_stack.push( it );
            it = it->node_begin();
        }
        while ( !it->empty() );
    }

    else
    {
        if ( !node_stack.empty() && it == node_stack.top()->node_end() )
        {
            it = node_stack.top();
            node_stack.pop();
        }

        else if ( node_stack.empty() && it == pTop_node->node_end() )
        {
            at_top = true;
        }
    }

    return *this;
}


template<typename stored_type, typename tree_type, typename container_type, typename base_iterator_type, typename pointer_type, typename reference_type>
tcl::post_order_descendant_node_iterator<stored_type, tree_type, container_type, base_iterator_type, pointer_type, reference_type>& tcl::post_order_descendant_node_iterator<stored_type, tree_type, container_type, base_iterator_type, pointer_type, reference_type>::operator --()
{
    if ( at_top )
    {
        at_top = false;
        typename container_type::const_reverse_iterator rit = pTop_node->children.rbegin();
        ++rit;
        it = base_iterator_type( rit.base(), pTop_node );
    }

    else if ( it == pTop_node->node_end() )
    {
        at_top = true;
    }

    else
    {
        if ( !it->empty() )
        {
            typename container_type::const_reverse_iterator rit = it->children.rbegin();
            node_stack.push( it );
            ++rit;
            it = base_iterator_type( rit.base(), &( *it ) );
        }

        else
        {
            if ( it != it->parent()->node_begin() )
            {
                --it;
            }

            else
            {
                while ( !node_stack.empty() && it == node_stack.top()->node_begin() )
                {
                    it = node_stack.top();
                    node_stack.pop();
                }

                --it;
            }
        }
    }

    return *this;
}


template<typename stored_type, typename tree_type, typename container_type, typename base_iterator_type, typename pointer_type, typename reference_type>
tcl::level_order_descendant_node_iterator<stored_type, tree_type, container_type, base_iterator_type, pointer_type, reference_type>& tcl::level_order_descendant_node_iterator<stored_type, tree_type, container_type, base_iterator_type, pointer_type, reference_type>::operator ++()
{
    if ( at_top )
    {
        at_top = false;
        it = pTop_node->node_begin();
        return *this;
    }

    const base_iterator_type it_end = it->parent()->node_end();
    node_queue.push( it );
    ++it;

    if ( it == it_end )
    {
        while ( !node_queue.empty() )
        {
            it = node_queue.front();
            node_queue.pop();

            if ( !it->empty() )
            {
                it = it->node_begin();
                break;
            }

            else if ( node_queue.empty() )
            {
                it = pTop_node->node_end();
                return *this;
            }
        }
    }

    return *this;
}
// 285 "descendant_node_iterator.h" 2
// 33 "associative_tree.h" 2
// 1 "reverse_iterator.h" 1
// 27 "reverse_iterator.h"




namespace tcl
{
    template<typename T, typename U, typename V, typename W, typename X, typename Y> class associative_reverse_iterator;
    template<typename T, typename U, typename V, typename W, typename X, typename Y> class const_sequential_reverse_iterator;
}



template<typename stored_type, typename tree_type, typename tree_pointer_type, typename container_type, typename pointer_type, typename reference_type>
class tcl::associative_reverse_iterator : public tcl::associative_iterator<stored_type, tree_type, tree_pointer_type, container_type, pointer_type, reference_type>
{
        typedef associative_iterator<stored_type, tree_type, tree_pointer_type, container_type, pointer_type, reference_type> associative_iterator_type;
    public:
        associative_reverse_iterator() : associative_iterator_type() {}
        explicit associative_reverse_iterator( const associative_iterator_type& _it ) : associative_iterator_type( _it ) {}

        associative_reverse_iterator( const associative_reverse_iterator<stored_type, tree_type, tree_type*, container_type, stored_type*, stored_type&>& src ) : associative_iterator_type( src ) {}

        reference_type operator*() const { associative_iterator_type tmp( *this ); return( *--tmp );}
        pointer_type operator->() const { associative_iterator_type tmp( *this ); --tmp; return tmp.operator ->();}
        associative_reverse_iterator& operator ++() { associative_iterator_type::operator --(); return *this;}
        associative_reverse_iterator operator ++( int ) { associative_reverse_iterator old( *this ); ++*this; return old;}
        associative_reverse_iterator& operator --() { associative_iterator_type::operator ++(); return *this;}
        associative_reverse_iterator operator --( int ) { associative_reverse_iterator old( *this ); --*this; return old;}

        tree_pointer_type node() const { associative_iterator_type tmp( *this ); --tmp; return tmp.node();}
        associative_iterator_type base() const { return associative_iterator_type( *this );}
};




template<typename stored_type, typename tree_type, typename tree_pointer_type, typename container_type, typename pointer_type, typename reference_type>
class tcl::sequential_reverse_iterator : public tcl::sequential_iterator<stored_type, tree_type, tree_pointer_type, container_type, pointer_type, reference_type>
{
        typedef sequential_iterator<stored_type, tree_type, tree_pointer_type, container_type, pointer_type, reference_type> sequential_iterator_type;
    public:
        sequential_reverse_iterator() : sequential_iterator_type() {}
        explicit sequential_reverse_iterator( const sequential_iterator_type& _it ) : sequential_iterator_type( _it ) {}

        sequential_reverse_iterator( const sequential_reverse_iterator<stored_type, tree_type, tree_type*, container_type, stored_type*, stored_type&>& src ) : sequential_iterator_type( src ) {}

        reference_type operator*() const { sequential_iterator_type tmp( *this ); return( *--tmp );}
        pointer_type operator->() const { sequential_iterator_type tmp( *this ); --tmp; return tmp.operator ->();}
        sequential_reverse_iterator& operator ++() { sequential_iterator_type::operator --(); return *this;}
        sequential_reverse_iterator operator ++( int ) { sequential_reverse_iterator old( *this ); ++*this; return old;}
        sequential_reverse_iterator& operator --() { sequential_iterator_type::operator ++(); return *this;}
        sequential_reverse_iterator operator --( int ) { sequential_reverse_iterator old( *this ); --*this; return old;}

        tree_pointer_type node() const { sequential_iterator_type tmp( *this ); --tmp; return tmp.node();}
        sequential_iterator_type base() const { return sequential_iterator_type( *this );}
};
// 34 "associative_tree.h" 2
// 1 "reverse_node_iterator.h" 1
// 27 "reverse_node_iterator.h"




namespace tcl
{
    template<typename T, typename U, typename V, typename W, typename X> class associative_reverse_node_iterator;
    template<typename T, typename U, typename V, typename W, typename X> class sequential_reverse_node_iterator;
}


template<typename stored_type, typename tree_type, typename container_type, typename pointer_type, typename reference_type>
class tcl::associative_reverse_node_iterator : public tcl::associative_node_iterator<stored_type, tree_type, container_type, pointer_type, reference_type>
{
        typedef associative_node_iterator<stored_type, tree_type, container_type, pointer_type, reference_type> associative_iterator_type;
    public:
        associative_reverse_node_iterator() : associative_iterator_type() {}
        explicit associative_reverse_node_iterator( const associative_iterator_type& _it ) : associative_iterator_type( _it ) {}

        reference_type operator*() const { associative_iterator_type tmp( *this ); return( *--tmp );}
        pointer_type operator->() const { associative_iterator_type tmp( *this ); --tmp; return tmp.operator ->();}
        associative_reverse_node_iterator& operator ++() { associative_iterator_type::operator --(); return *this;}
        associative_reverse_node_iterator operator ++( int ) { associative_reverse_node_iterator old( *this ); ++*this; return old;}
        associative_reverse_node_iterator& operator --() { associative_iterator_type::operator ++(); return *this;}
        associative_reverse_node_iterator operator --( int ) { associative_reverse_node_iterator old( *this ); --*this; return old;}

        associative_iterator_type base() const { return associative_iterator_type( *this );}
};




template<typename stored_type, typename tree_type, typename container_type, typename pointer_type, typename reference_type>
class tcl::sequential_reverse_node_iterator : public tcl::sequential_node_iterator<stored_type, tree_type, container_type, pointer_type, reference_type>
{
        typedef sequential_node_iterator<stored_type, tree_type, container_type, pointer_type, reference_type> sequential_iterator_type;
    public:
        sequential_reverse_node_iterator() : sequential_iterator_type() {}
        explicit sequential_reverse_node_iterator( const sequential_iterator_type& _it ) : sequential_iterator_type( _it ) {}

        reference_type operator*() const { sequential_iterator_type tmp( *this ); return( *--tmp );}
        pointer_type operator->() const { sequential_iterator_type tmp( *this ); --tmp; return tmp.operator ->();}
        sequential_reverse_node_iterator& operator ++() { sequential_iterator_type::operator --(); return *this;}
        sequential_reverse_node_iterator operator ++( int ) { sequential_reverse_node_iterator old( *this ); ++*this; return old;}
        sequential_reverse_node_iterator& operator --() { sequential_iterator_type::operator ++(); return *this;}
        sequential_reverse_node_iterator operator --( int ) { sequential_reverse_node_iterator old( *this ); --*this; return old;}

        sequential_iterator_type base() const { return sequential_iterator_type( *this );}
};
// 35 "associative_tree.h" 2


namespace tcl
{
    template<typename T, typename U, typename X> class associative_tree;


    template<typename T, typename U, typename V> bool operator == ( const associative_tree<T, U, V>& lhs, const associative_tree<T, U, V>& rhs );
    template<typename T, typename U, typename V> bool operator < ( const associative_tree<T, U, V>& lhs, const associative_tree<T, U, V>& rhs );
    template<typename T, typename U, typename V> bool operator != ( const associative_tree<T, U, V>& lhs, const associative_tree<T, U, V>& rhs ) { return !( lhs == rhs );}
    template<typename T, typename U, typename V> bool operator > ( const associative_tree<T, U, V>& lhs, const associative_tree<T, U, V>& rhs ) { return rhs < lhs;}
    template<typename T, typename U, typename V> bool operator <= ( const associative_tree<T, U, V>& lhs, const associative_tree<T, U, V>& rhs ) { return !( rhs < lhs );}
    template<typename T, typename U, typename V> bool operator >= ( const associative_tree<T, U, V>& lhs, const associative_tree<T, U, V>& rhs ) { return !( lhs < rhs );}
}





template< typename stored_type, typename tree_type, typename container_type >
class tcl::associative_tree : public basic_tree<stored_type, tree_type, container_type>
{
    protected:
        typedef basic_tree<stored_type, tree_type, container_type> basic_tree_type;
        explicit associative_tree( const stored_type& stored_obj ) : basic_tree_type( stored_obj ) {}
        virtual ~associative_tree() {}

    public:

        typedef associative_tree<stored_type, tree_type, container_type> associative_tree_type;
        typedef stored_type key_type;
        using basic_tree_type::size_type;


        typedef associative_iterator<stored_type, tree_type, const tree_type*, container_type, const stored_type*, const stored_type&> const_iterator;
        typedef associative_iterator<stored_type, tree_type, tree_type*, container_type, stored_type*, stored_type&> iterator;
        typedef associative_reverse_iterator<stored_type, tree_type, const tree_type*, container_type, const stored_type*, const stored_type&> const_reverse_iterator;
        typedef associative_reverse_iterator<stored_type, tree_type, tree_type*, container_type, stored_type*, stored_type&> reverse_iterator;
        typedef pre_order_descendant_iterator<stored_type, tree_type, const tree_type*, container_type, const_iterator, const stored_type*, const stored_type&> const_pre_order_iterator;
        typedef pre_order_descendant_iterator<stored_type, tree_type, tree_type*, container_type, iterator, stored_type*, stored_type&> pre_order_iterator;
        typedef post_order_descendant_iterator<stored_type, tree_type, const tree_type*, container_type, const_iterator, const stored_type*, const stored_type&> const_post_order_iterator;
        typedef post_order_descendant_iterator<stored_type, tree_type, tree_type*, container_type, iterator, stored_type*, stored_type&> post_order_iterator;
        typedef level_order_descendant_iterator<stored_type, tree_type, const tree_type*, container_type, const_iterator, const stored_type*, const stored_type&> const_level_order_iterator;
        typedef level_order_descendant_iterator<stored_type, tree_type, tree_type*, container_type, iterator, stored_type*, stored_type&> level_order_iterator;


        typedef associative_node_iterator<stored_type, tree_type, container_type, const tree_type*, const tree_type&> const_node_iterator;
        typedef associative_node_iterator<stored_type, tree_type, container_type, tree_type*, tree_type&> node_iterator;
        typedef associative_reverse_node_iterator<stored_type, tree_type, container_type, const tree_type*, const tree_type&> const_reverse_node_iterator;
        typedef associative_reverse_node_iterator<stored_type, tree_type, container_type, tree_type*, tree_type&> reverse_node_iterator;
        typedef pre_order_descendant_node_iterator<stored_type, tree_type, container_type, const_node_iterator, const tree_type*, const tree_type&> const_pre_order_node_iterator;
        typedef pre_order_descendant_node_iterator<stored_type, tree_type, container_type, node_iterator, tree_type*, tree_type&> pre_order_node_iterator;
        typedef post_order_descendant_node_iterator<stored_type, tree_type, container_type, const_node_iterator, const tree_type*, const tree_type&> const_post_order_node_iterator;
        typedef post_order_descendant_node_iterator<stored_type, tree_type, container_type, node_iterator, tree_type*, tree_type&> post_order_node_iterator;
        typedef level_order_descendant_node_iterator<stored_type, tree_type, container_type, const_node_iterator, const tree_type*, const tree_type&> const_level_order_node_iterator;
        typedef level_order_descendant_node_iterator<stored_type, tree_type, container_type, node_iterator, tree_type*, tree_type&> level_order_node_iterator;


        const_iterator begin() const { return const_iterator( basic_tree_type::children.begin(), this );}
        const_iterator end() const { return const_iterator( basic_tree_type::children.end(), this );}
        iterator begin() { return iterator( basic_tree_type::children.begin(), this );}
        iterator end() { return iterator( basic_tree_type::children.end(), this );}


        const_node_iterator node_begin() const { return const_node_iterator( basic_tree_type::children.begin(), this );}
        const_node_iterator node_end() const { return const_node_iterator( basic_tree_type::children.end(), this );}
        node_iterator node_begin() { return node_iterator( basic_tree_type::children.begin(), this );}
        node_iterator node_end() { return node_iterator( basic_tree_type::children.end(), this );}


        const_reverse_iterator rbegin() const {return const_reverse_iterator( end() );}
        const_reverse_iterator rend() const {return const_reverse_iterator( begin() );}
        reverse_iterator rbegin() {return reverse_iterator( end() );}
        reverse_iterator rend() { return reverse_iterator( begin() );}


        const_reverse_node_iterator node_rbegin() const {return const_reverse_node_iterator( node_end() );}
        const_reverse_node_iterator node_rend() const {return const_reverse_node_iterator( node_begin() );}
        reverse_node_iterator node_rbegin() {return reverse_node_iterator( node_end() );}
        reverse_node_iterator node_rend() { return reverse_node_iterator( node_begin() );}


        iterator find( const stored_type& value );
        const_iterator find( const stored_type& value ) const;
        bool erase( const stored_type& value );
        void erase( iterator it );
        void erase( iterator it_beg, iterator it_end );
        void clear();
        typename basic_tree_type::size_type count( const stored_type& value ) const;
        iterator lower_bound( const stored_type& value );
        const_iterator lower_bound( const stored_type& value ) const;
        iterator upper_bound( const stored_type& value );
        const_iterator upper_bound( const stored_type& value ) const;
        std::pair<iterator, iterator> equal_range( const stored_type& value )
        {
            tree_type node_obj( value );
            iterator lower_it( basic_tree_type::children.lower_bound( &node_obj ), this );
            iterator upper_it( basic_tree_type::children.upper_bound( &node_obj ), this );
            return std::make_pair( lower_it, upper_it );
        }
        std::pair<const_iterator, const_iterator> equal_range( const stored_type& value ) const
        {
            tree_type node_obj( value );
            const_iterator lower_it( basic_tree_type::children.lower_bound( &node_obj ), this );
            const_iterator upper_it( basic_tree_type::children.upper_bound( &node_obj ), this );
            return std::make_pair( lower_it, upper_it );
        }

    protected:
        iterator insert( const stored_type& value, tree_type* parent ) { return insert( end(), value, parent );}
        iterator insert( const const_iterator& pos, const stored_type& value, tree_type* parent );
        iterator insert( const tree_type& tree_obj, tree_type* parent ) { return insert( end(), tree_obj, parent );}
        iterator insert( const const_iterator pos, const tree_type& tree_obj, tree_type* parent );
        void set( const tree_type& tree_obj, tree_type* parent );
        // 178 "associative_tree.h"
        template<typename T, typename U, typename V, typename W, typename X, typename Y, typename Z> friend class pre_order_descendant_iterator;
        template<typename T, typename U, typename V, typename W, typename X, typename Y, typename Z> friend class post_order_descendant_iterator;
        template<typename T, typename U, typename V, typename W, typename X, typename Y, typename Z> friend class level_order_descendant_iterator;
        template<typename T, typename U, typename V, typename W, typename X, typename Y> friend class pre_order_descendant_node_iterator;
        template<typename T, typename U, typename V, typename W, typename X, typename Y> friend class post_order_descendant_node_iterator;
        template<typename T, typename U, typename V, typename W, typename X, typename Y> friend class level_order_descendant_node_iterator;
        friend bool operator ==<> ( const associative_tree_type& lhs, const associative_tree_type& rhs );
        friend bool operator < <> ( const associative_tree_type& lhs, const associative_tree_type& rhs );

};

// 1 "associative_tree.inl" 1
// 29 "associative_tree.inl"
template< typename stored_type, typename tree_type, typename container_type >
typename tcl::associative_tree<stored_type, tree_type, container_type>::iterator
tcl::associative_tree<stored_type, tree_type, container_type>::find( const stored_type& value )
{
    tree_type node_obj( value );
    iterator it( basic_tree_type::children.find( &node_obj ), this );
    return it;
}


template< typename stored_type, typename tree_type, typename container_type >
typename tcl::associative_tree<stored_type, tree_type, container_type>::const_iterator
tcl::associative_tree<stored_type, tree_type, container_type>::find( const stored_type& value ) const
{
    tree_type node_obj( value );
    const_iterator it( basic_tree_type::children.find( &node_obj ), this );
    return it;
}



template< typename stored_type, typename tree_type, typename container_type >
bool tcl::associative_tree<stored_type, tree_type, container_type>::
erase( const stored_type& value )
{
    bool erased_nodes = false;
    tree_type node_obj( value );
    typename container_type::iterator it = basic_tree_type::children.find( &node_obj );

    while ( it != basic_tree_type::children.end() )
    {
        deallocate_tree_type( *it );
        basic_tree_type::children.erase( it );
        it = basic_tree_type::children.find( &node_obj );
        erased_nodes = true;
    }

    return erased_nodes;
}


template< typename stored_type, typename tree_type, typename container_type >
typename tcl::basic_tree<stored_type, tree_type, container_type>::size_type
tcl::associative_tree<stored_type, tree_type, container_type>::count( const stored_type& value ) const
{
    const_iterator it = find( value );
    const_iterator it_end = end();
    typename basic_tree_type::size_type cnt = 0;

    while ( it != it_end && !( *it < value || value < *it ) )
    {
        ++cnt;
        ++it;
    }

    return cnt;
}


template< typename stored_type, typename tree_type, typename container_type>
typename tcl::associative_tree<stored_type, tree_type, container_type>::iterator
tcl::associative_tree<stored_type, tree_type, container_type>::lower_bound( const stored_type& value )
{
    tree_type node_obj( value );
    iterator it( basic_tree_type::children.lower_bound( &node_obj ), this );
    return it;
}


template< typename stored_type, typename tree_type, typename container_type>
typename tcl::associative_tree<stored_type, tree_type, container_type>::const_iterator
tcl::associative_tree<stored_type, tree_type, container_type>::lower_bound( const stored_type& value ) const
{
    tree_type node_obj( value );
    const_iterator it( basic_tree_type::children.lower_bound( &node_obj ), this );
    return it;
}


template< typename stored_type, typename tree_type, typename container_type>
typename tcl::associative_tree<stored_type, tree_type, container_type>::iterator
tcl::associative_tree<stored_type, tree_type, container_type>::upper_bound( const stored_type& value )
{
    tree_type node_obj( value );
    iterator it( basic_tree_type::children.upper_bound( &node_obj ), this );
    return it;
}


template< typename stored_type, typename tree_type, typename container_type>
typename tcl::associative_tree<stored_type, tree_type, container_type>::const_iterator
tcl::associative_tree<stored_type, tree_type, container_type>::upper_bound( const stored_type& value ) const
{
    tree_type node_obj( value );
    const_iterator it( basic_tree_type::children.upper_bound( &node_obj ), this );
    return it;
}


template< typename stored_type, typename tree_type, typename container_type>
typename tcl::associative_tree<stored_type, tree_type, container_type>::iterator
tcl::associative_tree<stored_type, tree_type, container_type>::insert( const const_iterator& pos, const stored_type& value, tree_type* pParent )
{
    tree_type* pNew_node;
    basic_tree_type::allocate_tree_type( pNew_node, tree_type( value ) );
    pNew_node->set_parent( pParent );
    const typename basic_tree_type::size_type sz = basic_tree_type::children.size();
    typename container_type::iterator children_pos;

    if ( pos == pParent->end() )
    {
        children_pos = basic_tree_type::children.end();
    }

    else
    {
        children_pos = basic_tree_type::children.begin();
        typename container_type::const_iterator const_children_pos = basic_tree_type::children.begin();

        while ( const_children_pos != pos.it && const_children_pos != basic_tree_type::children.end() )
        {
            ++children_pos;
            ++const_children_pos;
        }
    }

    const typename container_type::iterator it = basic_tree_type::children.insert( children_pos, pNew_node );

    if ( sz == basic_tree_type::children.size() )
    {
        basic_tree_type::deallocate_tree_type( pNew_node );
        return iterator( basic_tree_type::children.end(), this );
    }

    return iterator( it, this );
}


template< typename stored_type, typename tree_type, typename container_type>
typename tcl::associative_tree<stored_type, tree_type, container_type>::iterator
tcl::associative_tree<stored_type, tree_type, container_type>::insert( const const_iterator pos, const tree_type& tree_obj, tree_type* pParent )
{
    iterator base_it = pParent->insert( pos, *tree_obj.get() );

    if ( base_it != pParent->end() )
    {
        const_iterator it = tree_obj.begin();
        const const_iterator it_end = tree_obj.end();

        for ( ; it != it_end; ++it )
        { base_it.node()->insert( *it.node() ); }
    }

    return base_it;
}


template< typename stored_type, typename tree_type, typename container_type>
void tcl::associative_tree<stored_type, tree_type, container_type>::set( const tree_type& tree_obj, tree_type* pParent )
{
    set( *tree_obj.get() );
    const_iterator it = tree_obj.begin();
    const const_iterator it_end = tree_obj.end();

    for ( ; it != it_end; ++it )
    {
        insert( *it.node(), pParent );
    }
}


template< typename stored_type, typename tree_type, typename container_type>
void tcl::associative_tree<stored_type, tree_type, container_type>::clear()
{
    iterator it = begin();
    const iterator it_end = end();

    for ( ; it != it_end; ++it )
    {
        basic_tree_type::deallocate_tree_type( it.node() );
    }

    basic_tree_type::children.clear();
}


template< typename stored_type, typename tree_type, typename container_type>
void tcl::associative_tree<stored_type, tree_type, container_type>::erase( iterator it )
{
    if ( it.pParent != this )
    { return; }

    it.node()->clear();
    deallocate_tree_type( it.node() );
    const iterator beg_it = begin();
    typename container_type::iterator pos_it = basic_tree_type::children.begin();

    for ( ; it != beg_it; --it, ++pos_it ) ;

    basic_tree_type::children.erase( pos_it );
}


template< typename stored_type, typename tree_type, typename container_type>
void tcl::associative_tree<stored_type, tree_type, container_type>::erase( iterator it_beg, iterator it_end )
{
    while ( it_beg != it_end )
    {
        erase( it_beg++ );
    }
}



template<typename stored_type, typename tree_type, typename container_type>
bool tcl::operator == ( const associative_tree<stored_type, tree_type, container_type>& lhs, const associative_tree<stored_type, tree_type, container_type>& rhs )
{
    if ( ( *lhs.get() < *rhs.get() ) || ( *rhs.get() < *lhs.get() ) )
    { return false; }

    typename associative_tree<stored_type, tree_type, container_type>::const_iterator lhs_it = lhs.begin();
    const typename associative_tree<stored_type, tree_type, container_type>::const_iterator lhs_end = lhs.end();
    typename associative_tree<stored_type, tree_type, container_type>::const_iterator rhs_it = rhs.begin();
    const typename associative_tree<stored_type, tree_type, container_type>::const_iterator rhs_end = rhs.end();

    for ( ; lhs_it != lhs_end && rhs_it != rhs_end; ++lhs_it, ++rhs_it )
    {
        if ( *lhs_it.node() != *rhs_it.node() )
        {
            return false;
        }
    }

    if ( lhs_it != lhs.end() || rhs_it != rhs.end() )
    { return false; }

    return true;
}



template<typename stored_type, typename tree_type, typename container_type>
bool tcl::operator < ( const associative_tree<stored_type, tree_type, container_type>& lhs, const associative_tree<stored_type, tree_type, container_type>& rhs )
{
    if ( *lhs.get() < *rhs.get() )
    { return true; }

    typename associative_tree<stored_type, tree_type, container_type>::const_iterator lhs_it = lhs.begin();
    const typename associative_tree<stored_type, tree_type, container_type>::const_iterator lhs_end = lhs.end();
    typename associative_tree<stored_type, tree_type, container_type>::const_iterator rhs_it = rhs.begin();
    const typename associative_tree<stored_type, tree_type, container_type>::const_iterator rhs_end = rhs.end();

    for ( ; lhs_it != lhs_end && rhs_it != rhs_end; ++lhs_it, ++rhs_it )
    {
        if ( *lhs_it.node() < *rhs_it.node() )
        {
            return true;
        }
    }

    if ( lhs.size() != rhs.size() )
    {
        return lhs.size() < rhs.size();
    }

    return false;
}
// 190 "associative_tree.h" 2
// 29 "Tree.h" 2


namespace tcl
{

    template<typename stored_type, typename node_compare_type > class tree;


    template<typename stored_type, typename node_compare_type >
    struct tree_deref_less : public std::binary_function<const tree<stored_type, node_compare_type>*, const tree<stored_type, node_compare_type>*, bool>
    {
        bool operator()( const tree<stored_type, node_compare_type>* lhs, const tree<stored_type, node_compare_type>* rhs ) const
        {
            return node_compare_type()( *lhs->get(), *rhs->get() );
        }
    };
}






template<typename stored_type, typename node_compare_type = std::less<stored_type> >
class tcl::tree : public tcl::associative_tree<stored_type, tcl::tree<stored_type, node_compare_type>, std::set<tcl::tree<stored_type, node_compare_type>*, tcl::tree_deref_less<stored_type, node_compare_type> > >
{
    public:

        typedef tree<stored_type, node_compare_type> tree_type;
        typedef tree_deref_less<stored_type, node_compare_type> key_compare;
        typedef tree_deref_less<stored_type, node_compare_type> value_compare;
        typedef std::set<tree_type*, key_compare> container_type;
        typedef basic_tree<stored_type, tree_type, container_type> basic_tree_type;
        typedef associative_tree<stored_type, tree_type, container_type> associative_tree_type;


        explicit tree( const stored_type& value = stored_type() ) : associative_tree_type( value ) {}
        template<typename iterator_type> tree( iterator_type it_beg, iterator_type it_end, const stored_type& value = stored_type() ) : associative_tree_type( value )
        {
            while ( it_beg != it_end )
            {
                insert( *it_beg );
                ++it_beg;
            }
        }
        tree( const tree_type& rhs );
        ~tree() { associative_tree_type::clear();}


        tree_type& operator = ( const tree_type& rhs );


    public:
        typename associative_tree_type::iterator insert( const stored_type& value ) { return associative_tree_type::insert( value, this );}
        typename associative_tree_type::iterator insert( const typename associative_tree_type::const_iterator pos, const stored_type& value ) { return associative_tree_type::insert( pos, value, this );}
        typename associative_tree_type::iterator insert( const tree_type& tree_obj ) { return associative_tree_type::insert( tree_obj, this );}
        typename associative_tree_type::iterator insert( const typename associative_tree_type::const_iterator pos, const tree_type& tree_obj ) { return associative_tree_type::insert( pos, tree_obj, this );}

        template<typename iterator_type> void insert( iterator_type it_beg, iterator_type it_end ) { while ( it_beg != it_end ) { insert( *it_beg++ ); }}

        void swap( tree_type& rhs );


        typedef typename associative_tree_type::post_order_iterator post_order_iterator_type;
        typedef typename associative_tree_type::const_post_order_iterator const_post_order_iterator_type;
        typedef typename associative_tree_type::pre_order_iterator pre_order_iterator_type;
        typedef typename associative_tree_type::const_pre_order_iterator const_pre_order_iterator_type;
        typedef typename associative_tree_type::level_order_iterator level_order_iterator_type;
        typedef typename associative_tree_type::const_level_order_iterator const_level_order_iterator_type;

        pre_order_iterator_type pre_order_begin() { return pre_order_iterator_type( this, true );}
        pre_order_iterator_type pre_order_end() { return pre_order_iterator_type( this, false );}
        const_pre_order_iterator_type pre_order_begin() const { return const_pre_order_iterator_type( this, true );}
        const_pre_order_iterator_type pre_order_end() const { return const_pre_order_iterator_type( this, false );}
        post_order_iterator_type post_order_begin() { return post_order_iterator_type( this, true );}
        post_order_iterator_type post_order_end() { return post_order_iterator_type( this, false );}
        const_post_order_iterator_type post_order_begin() const { return const_post_order_iterator_type( this, true );}
        const_post_order_iterator_type post_order_end() const { return const_post_order_iterator_type( this, false );}
        level_order_iterator_type level_order_begin() { return level_order_iterator_type( this, true );}
        level_order_iterator_type level_order_end() { return level_order_iterator_type( this, false );}
        const_level_order_iterator_type level_order_begin() const { return const_level_order_iterator_type( this, true );}
        const_level_order_iterator_type level_order_end() const { return const_level_order_iterator_type( this, false );}


        typedef typename associative_tree_type::pre_order_node_iterator pre_order_node_iterator_type;
        typedef typename associative_tree_type::const_pre_order_node_iterator const_pre_order_node_iterator_type;
        typedef typename associative_tree_type::post_order_node_iterator post_order_node_iterator_type;
        typedef typename associative_tree_type::const_post_order_node_iterator const_post_order_node_iterator_type;
        typedef typename associative_tree_type::level_order_node_iterator level_order_node_iterator_type;
        typedef typename associative_tree_type::const_level_order_node_iterator const_level_order_node_iterator_type;

        pre_order_node_iterator_type pre_order_node_begin() { return pre_order_node_iterator_type( this, true );}
        pre_order_node_iterator_type pre_order_node_end() { return pre_order_node_iterator_type( this, false );}
        const_pre_order_node_iterator_type pre_order_node_begin() const { return const_pre_order_node_iterator_type( this, true );}
        const_pre_order_node_iterator_type pre_order_node_end() const { return const_pre_order_node_iterator_type( this, false );}
        post_order_node_iterator_type post_order_node_begin() { return post_order_node_iterator_type( this, true );}
        post_order_node_iterator_type post_order_node_end() { return post_order_node_iterator_type( this, false );}
        const_post_order_node_iterator_type post_order_node_begin() const { return const_post_order_node_iterator_type( this, true );}
        const_post_order_node_iterator_type post_order_node_end() const { return const_post_order_node_iterator_type( this, false );}
        level_order_node_iterator_type level_order_node_begin() { return level_order_node_iterator_type( this, true );}
        level_order_node_iterator_type level_order_node_end() { return level_order_node_iterator_type( this, false );}
        const_level_order_node_iterator_type level_order_node_begin() const { return const_level_order_node_iterator_type( this, true );}
        const_level_order_node_iterator_type level_order_node_end() const { return const_level_order_node_iterator_type( this, false );}





        template<typename T, typename U, typename V> friend class basic_tree;

};// 1 "Tree.h"
// 1 "<command-line>"
// 1 "/usr/include/stdc-predef.h" 1 3 4
// 1 "<command-line>" 2
// 1 "Tree.h"
// 27 "Tree.h"

// 1 "associative_tree.h" 1
// 27 "associative_tree.h"

// 1 "basic_tree.h" 1
// 27 "basic_tree.h"







namespace tcl
{
    template<typename T, typename U, typename V> class basic_tree;
}





template< typename stored_type, typename tree_type, typename container_type >
class tcl::basic_tree
{
    public:

        typedef basic_tree<stored_type, tree_type, container_type> basic_tree_type;
        typedef stored_type* ( *tClone_fcn )( const stored_type& );
        typedef stored_type value_type;
        typedef stored_type& reference;
        typedef const stored_type& const_reference;
        typedef size_t size_type;
        typedef std::allocator<stored_type> allocator_type;
        typedef typename allocator_type::difference_type difference_type;

    protected:

        basic_tree() : pElement( 0 ), pParent_node( 0 ) {}
        explicit basic_tree( const stored_type& value );
        basic_tree( const basic_tree_type& rhs );
        virtual ~basic_tree();

    public:

        const stored_type* get() const { return pElement;}
        stored_type* get() { return pElement;}
        bool is_root() const { return pParent_node == 0;}
        size_type size() const { return children.size();}
        size_type max_size() const { return( std::numeric_limits<int>().max )();}
        bool empty() const { return children.empty();}
        tree_type* parent() { return pParent_node;}
        const tree_type* parent() const { return pParent_node;}
        static void set_clone( const tClone_fcn& fcn ) { pClone_fcn = fcn;}


    protected:
        void set_parent( tree_type* pParent ) { pParent_node = pParent;}
        basic_tree_type& operator = ( const basic_tree_type& rhs );
        void set( const stored_type& stored_obj );
        void allocate_stored_type( stored_type*& element_ptr, const stored_type& value )
        { element_ptr = stored_type_allocator.allocate( 1, 0 ); stored_type_allocator.construct( element_ptr, value );}
        void deallocate_stored_type( stored_type* element_ptr )
        { stored_type_allocator.destroy( element_ptr ); stored_type_allocator.deallocate( element_ptr, 1 );}
        void allocate_tree_type( tree_type*& tree_ptr, const tree_type& tree_obj )
        { tree_ptr = tree_type_allocator.allocate( 1, 0 ); tree_type_allocator.construct( tree_ptr, tree_obj );}
        void deallocate_tree_type( tree_type* tree_ptr )
        { tree_type_allocator.destroy( tree_ptr ); tree_type_allocator.deallocate( tree_ptr, 1 );}


    protected:
        container_type children;
    private:
        stored_type* pElement;
        mutable tree_type* pParent_node;
        static tClone_fcn pClone_fcn;
        std::allocator<stored_type> stored_type_allocator;
        std::allocator<tree_type> tree_type_allocator;
};

// 1 "basic_tree.inl" 1
// 29 "basic_tree.inl"
template< typename stored_type, typename tree_type, typename container_type >
typename tcl::basic_tree<stored_type, tree_type, container_type>::tClone_fcn
tcl::basic_tree<stored_type, tree_type, container_type>::pClone_fcn = 0;


template< typename stored_type, typename tree_type, typename container_type >
tcl::basic_tree<stored_type, tree_type, container_type>::basic_tree( const stored_type& value )
    : children( container_type() ), pElement( 0 ), pParent_node( 0 ),
      stored_type_allocator( std::allocator<stored_type>() ), tree_type_allocator( std::allocator<tree_type>() )
{
    if ( pClone_fcn )
    { pElement = pClone_fcn( value ); }

    else
    { allocate_stored_type( pElement, value ); }
}



template< typename stored_type, typename tree_type, typename container_type >
tcl::basic_tree<stored_type, tree_type, container_type>::basic_tree( const basic_tree_type& rhs )
    : children( container_type() ), pElement( 0 ), pParent_node( 0 ),
      stored_type_allocator( std::allocator<stored_type>() ), tree_type_allocator( std::allocator<tree_type>() )
{
    pParent_node = 0;
    set( *rhs.get() );
}


template< typename stored_type, typename tree_type, typename container_type >
tcl::basic_tree<stored_type, tree_type, container_type>& tcl::basic_tree<stored_type, tree_type, container_type>::operator = ( const basic_tree_type& rhs )
{
    if ( &rhs == this )
    { return *this; }

    set( *rhs.get() );
    return *this;
}


template< typename stored_type, typename tree_type, typename container_type >
tcl::basic_tree<stored_type, tree_type, container_type>::~basic_tree()
{
    deallocate_stored_type( pElement );
}




template< typename stored_type, typename tree_type, typename container_type >
void tcl::basic_tree<stored_type, tree_type, container_type>::set( const stored_type& value )
{
    if ( pElement )
    { deallocate_stored_type( pElement ); }

    if ( pClone_fcn )
    { pElement = pClone_fcn( value ); }

    else
    { allocate_stored_type( pElement, value ); }
}
// 102 "basic_tree.h" 2
// 29 "associative_tree.h" 2
// 1 "child_iterator.h" 1
// 27 "child_iterator.h"







namespace tcl
{

    template<typename T, typename U, typename V> class basic_tree;
    template<typename T> class sequential_tree;
    template<typename T, typename U, typename V> class associative_tree;
    template<typename T, typename U, typename V, typename W, typename X, typename Y> class associative_reverse_iterator;
    template<typename T, typename U, typename V, typename W, typename X, typename Y> class sequential_reverse_iterator;
    template<typename T, typename U, typename V, typename W, typename X, typename Y, typename Z> class pre_order_descendant_iterator;
    template<typename T, typename U, typename V, typename W, typename X, typename Y, typename Z> class post_order_descendant_iterator;

    template<typename T, typename U, typename V, typename W, typename X, typename Y> class associative_iterator;
    template<typename T, typename U, typename V, typename W, typename X, typename Y> class sequential_iterator;

    template<typename ST, typename TT, typename TPT1, typename CT, typename PT1, typename RT1, typename TPT2, typename PT2, typename RT2>
    void associative_it_init( associative_iterator<ST, TT, TPT1, CT, PT1, RT1>* dest, const associative_iterator<ST, TT, TPT2, CT, PT2, RT2>& src ) { dest->it = src.it; dest->pParent = src.pParent; }

    template<typename ST, typename TT, typename TPT1, typename CT, typename PT1, typename RT1, typename TPT2, typename PT2, typename RT2>
    bool associative_it_eq( const associative_iterator<ST, TT, TPT1, CT, PT1, RT1>* lhs, const associative_iterator<ST, TT, TPT2, CT, PT2, RT2>& rhs ) { return lhs->pParent == rhs.pParent && lhs->it == rhs.it; }

    template<typename ST, typename TT, typename TPT1, typename CT, typename PT1, typename RT1, typename TPT2, typename PT2, typename RT2>
    void sequential_it_init( sequential_iterator<ST, TT, TPT1, CT, PT1, RT1>* dest, const sequential_iterator<ST, TT, TPT2, CT, PT2, RT2>& src ) { dest->it = src.it; dest->pParent = src.pParent; }

    template<typename ST, typename TT, typename TPT1, typename CT, typename PT1, typename RT1, typename TPT2, typename PT2, typename RT2>
    bool sequential_it_eq( const sequential_iterator<ST, TT, TPT1, CT, PT1, RT1>* lhs, const sequential_iterator<ST, TT, TPT2, CT, PT2, RT2>& rhs ) { return lhs->pParent == rhs.pParent && lhs->it == rhs.it; }

    template<typename ST, typename TT, typename TPT1, typename CT, typename PT1, typename RT1, typename TPT2, typename PT2, typename RT2>
    bool sequential_it_less( const sequential_iterator<ST, TT, TPT1, CT, PT1, RT1>* lhs, const sequential_iterator<ST, TT, TPT2, CT, PT2, RT2>& rhs ) { return lhs->pParent == rhs.pParent && lhs->it < rhs.it; }
}






template<typename stored_type, typename tree_type, typename tree_pointer_type, typename container_type, typename pointer_type, typename reference_type>
class tcl::associative_iterator



    : public std::iterator<std::bidirectional_iterator_tag, stored_type, ptrdiff_t, pointer_type, reference_type>

{
    public:

        associative_iterator() : pParent( 0 ) {}

        associative_iterator( const associative_iterator<stored_type, tree_type, tree_type*, container_type, stored_type*, stored_type&>& src ) { associative_it_init( this, src ); }
        virtual ~associative_iterator() {}
    protected:



        explicit associative_iterator( const typename container_type::const_iterator& iter, const associative_tree<stored_type, tree_type, container_type>* pCalled_node ) : it( iter ), pParent( pCalled_node ) {}


    public:

        reference_type operator*() const { return const_cast<reference_type>( *( *it )->get() );}
        pointer_type operator->() const { return const_cast<pointer_type>( ( *it )->get() );}

        associative_iterator& operator ++() { ++it; return *this;}
        associative_iterator operator ++( int ) { associative_iterator old( *this ); ++*this; return old;}
        associative_iterator& operator --() { --it; return *this;}
        associative_iterator operator --( int ) { associative_iterator old( *this ); --*this; return old;}


        tree_pointer_type node() const { return const_cast<tree_pointer_type>( *it );}


        bool operator == ( const associative_iterator<stored_type, tree_type, const tree_type*, container_type, const stored_type*, const stored_type&>& rhs ) const { return associative_it_eq( this, rhs ); }
        bool operator != ( const associative_iterator<stored_type, tree_type, const tree_type*, container_type, const stored_type*, const stored_type&>& rhs ) const { return !( *this == rhs ); }

    protected:

        typename container_type::const_iterator it;
        const associative_tree<stored_type, tree_type, container_type>* pParent;
        // 123 "child_iterator.h"
        template<typename T, typename U, typename V> friend class basic_tree;
        template<typename T, typename U, typename V> friend class associative_tree;
        template<typename T, typename U, typename V, typename W, typename X, typename Y, typename Z> friend class pre_order_descendant_iterator;
        template<typename T, typename U, typename V, typename W, typename X, typename Y, typename Z> friend class post_order_descendant_iterator;
        friend void associative_it_init<>( associative_iterator<stored_type, tree_type, const tree_type*, container_type, const stored_type*, const stored_type&>*, const associative_iterator<stored_type, tree_type, tree_type*, container_type, stored_type*, stored_type&>& );
        friend void associative_it_init<>( associative_iterator<stored_type, tree_type, tree_type*, container_type, stored_type*, stored_type&>*, const associative_iterator<stored_type, tree_type, tree_type*, container_type, stored_type*, stored_type&>& );
        friend bool associative_it_eq<>( const associative_iterator<stored_type, tree_type, const tree_type*, container_type, const stored_type*, const stored_type&>*, const associative_iterator<stored_type, tree_type, const tree_type*, container_type, const stored_type*, const stored_type&>& );
        friend bool associative_it_eq<>( const associative_iterator<stored_type, tree_type, tree_type*, container_type, stored_type*, stored_type&>*, const associative_iterator<stored_type, tree_type, const tree_type*, container_type, const stored_type*, const stored_type&>& );

};






template<typename stored_type, typename tree_type, typename tree_pointer_type, typename container_type, typename pointer_type, typename reference_type>
class tcl::sequential_iterator



    : public std::iterator<std::random_access_iterator_tag, stored_type, ptrdiff_t, pointer_type, reference_type>

{
    public:

        sequential_iterator() : pParent( 0 ) {}

        sequential_iterator( const sequential_iterator<stored_type, tree_type, tree_type*, container_type, stored_type*, stored_type&>& src ) { sequential_it_init( this, src );}
        virtual ~sequential_iterator() {}
    protected:



        explicit sequential_iterator( typename container_type::const_iterator iter, const tree_type* pCalled_node ) : it( iter ), pParent( pCalled_node ) {}


    public:

        typedef size_t size_type;



        typedef typename std::iterator_traits<sequential_iterator>::difference_type difference_type;



        reference_type operator*() const { return const_cast<reference_type>( *( *it )->get() );}
        pointer_type operator->() const { return const_cast<pointer_type>( ( *it )->get() );}

        sequential_iterator& operator ++() { ++it; return *this;}
        sequential_iterator operator ++( int ) { sequential_iterator old( *this ); ++*this; return old;}
        sequential_iterator& operator --() { --it; return *this;}
        sequential_iterator operator --( int ) { sequential_iterator old( *this ); --*this; return old;}
        sequential_iterator& operator +=( difference_type n ) { it += n; return *this;}
        sequential_iterator& operator -=( difference_type n ) { it -= n; return *this;}
        difference_type operator -( const sequential_iterator& rhs ) const { return it - rhs.it;}
        tree_pointer_type node() const { return const_cast<tree_pointer_type>( *it );}


        bool operator == ( const sequential_iterator<stored_type, tree_type, const tree_type*, container_type, const stored_type*, const stored_type&>& rhs ) const { return sequential_it_eq( this, rhs ); }
        bool operator != ( const sequential_iterator<stored_type, tree_type, const tree_type*, container_type, const stored_type*, const stored_type&>& rhs ) const { return !( *this == rhs ); }
        bool operator < ( const sequential_iterator<stored_type, tree_type, const tree_type*, container_type, const stored_type*, const stored_type&>& rhs ) const { return sequential_it_less( this, rhs ); }
        bool operator <= ( const sequential_iterator<stored_type, tree_type, const tree_type*, container_type, const stored_type*, const stored_type&>& rhs ) const { return *this < rhs || *this == rhs; }
        bool operator > ( const sequential_iterator<stored_type, tree_type, const tree_type*, container_type, const stored_type*, const stored_type&>& rhs ) const { return !( *this <= rhs ); }
        bool operator >= ( const sequential_iterator<stored_type, tree_type, const tree_type*, container_type, const stored_type*, const stored_type&>& rhs ) const { return !( *this < rhs ); }

    protected:

        typename container_type::const_iterator it;
        const tree_type* pParent;
        // 207 "child_iterator.h"
        template<typename T> friend class sequential_tree;
        template<typename T, typename U, typename V, typename W, typename X, typename Y, typename Z> friend class pre_order_descendant_iterator;
        template<typename T, typename U, typename V, typename W, typename X, typename Y, typename Z> friend class post_order_descendant_iterator;
        friend void sequential_it_init<>( sequential_iterator<stored_type, tree_type, const tree_type*, container_type, const stored_type*, const stored_type&>*, const sequential_iterator<stored_type, tree_type, tree_type*, container_type, stored_type*, stored_type&>& );
        friend void sequential_it_init<>( sequential_iterator<stored_type, tree_type, tree_type*, container_type, stored_type*, stored_type&>*, const sequential_iterator<stored_type, tree_type, tree_type*, container_type, stored_type*, stored_type&>& );
        friend bool sequential_it_eq<>( const sequential_iterator<stored_type, tree_type, const tree_type*, container_type, const stored_type*, const stored_type&>*, const sequential_iterator<stored_type, tree_type, const tree_type*, container_type, const stored_type*, const stored_type&>& );
        friend bool sequential_it_eq<>( const sequential_iterator<stored_type, tree_type, tree_type*, container_type, stored_type*, stored_type&>*, const sequential_iterator<stored_type, tree_type, const tree_type*, container_type, const stored_type*, const stored_type&>& );
        friend bool sequential_it_less<>( const sequential_iterator<stored_type, tree_type, const tree_type*, container_type, const stored_type*, const stored_type&>*, const sequential_iterator<stored_type, tree_type, const tree_type*, container_type, const stored_type*, const stored_type&>& );
        friend bool sequential_it_less<>( const sequential_iterator<stored_type, tree_type, tree_type*, container_type, stored_type*, stored_type&>*, const sequential_iterator<stored_type, tree_type, const tree_type*, container_type, const stored_type*, const stored_type&>& );

};
// 30 "associative_tree.h" 2
// 1 "child_node_iterator.h" 1
// 27 "child_node_iterator.h"







namespace tcl
{

    template<typename T, typename U, typename V> class basic_tree;
    template<typename T> class sequential_tree;
    template<typename T, typename U, typename V> class associative_tree;
    template<typename T, typename U, typename V, typename W, typename X, typename Z> class associative_reverse_iterator;
    template<typename T, typename U, typename V, typename W, typename X, typename Z> class sequential_reverse_iterator;
    template<typename T, typename U, typename V, typename W, typename X> class associative_reverse_node_iterator;
    template<typename T, typename U, typename V, typename W, typename X> class sequential_reverse_node_iterator;

    template<typename T, typename U, typename V, typename W, typename X> class associative_node_iterator;
    template<typename T, typename U, typename V, typename W, typename X> class sequential_node_iterator;
    template<typename T, typename U, typename V, typename W, typename X, typename Y> class pre_order_descendant_node_iterator;
    template<typename T, typename U, typename V, typename W, typename X, typename Y> class post_order_descendant_node_iterator;

    template<typename ST, typename TT, typename CT, typename PT1, typename RT1, typename PT2, typename RT2>
    void associative_node_it_init( associative_node_iterator<ST, TT, CT, PT1, RT1>* dest, const associative_node_iterator<ST, TT, CT, PT2, RT2>& src ) { dest->it = src.it; dest->pParent = src.pParent; }

    template<typename ST, typename TT, typename CT, typename PT1, typename RT1, typename PT2, typename RT2>
    bool associative_node_it_eq( const associative_node_iterator<ST, TT, CT, PT1, RT1>* lhs, const associative_node_iterator<ST, TT, CT, PT2, RT2>& rhs ) { return lhs->pParent == rhs.pParent && lhs->it == rhs.it; }

    template<typename ST, typename TT, typename CT, typename PT1, typename RT1, typename PT2, typename RT2>
    void sequential_node_it_init( sequential_node_iterator<ST, TT, CT, PT1, RT1>* dest, const sequential_node_iterator<ST, TT, CT, PT2, RT2>& src ) { dest->it = src.it; dest->pParent = src.pParent; }

    template<typename ST, typename TT, typename CT, typename PT1, typename RT1, typename PT2, typename RT2>
    bool sequential_node_it_eq( const sequential_node_iterator<ST, TT, CT, PT1, RT1>* lhs, const sequential_node_iterator<ST, TT, CT, PT2, RT2>& rhs ) { return lhs->pParent == rhs.pParent && lhs->it == rhs.it; }

    template<typename ST, typename TT, typename CT, typename PT1, typename RT1, typename PT2, typename RT2>
    bool sequential_node_it_less( const sequential_node_iterator<ST, TT, CT, PT1, RT1>* lhs, const sequential_node_iterator<ST, TT, CT, PT2, RT2>& rhs ) { return lhs->pParent == rhs.pParent && lhs->it < rhs.it; }
}







template<typename stored_type, typename tree_type, typename container_type, typename pointer_type, typename reference_type>
class tcl::associative_node_iterator



    : public std::iterator<std::bidirectional_iterator_tag, tree_type, ptrdiff_t, pointer_type, reference_type>

{
    public:

        associative_node_iterator() : pParent( 0 ) {}

        associative_node_iterator( const associative_node_iterator<stored_type, tree_type, container_type, tree_type*, tree_type&>& src ) { associative_node_it_init( this, src ); }
    protected:



        explicit associative_node_iterator( const typename container_type::const_iterator& iter, const associative_tree<stored_type, tree_type, container_type>* pCalled_node ) : it( iter ), pParent( pCalled_node ) {}


    public:

        reference_type operator*() const { return const_cast<reference_type>( *( *it ) );}
        pointer_type operator->() const { return const_cast<pointer_type>( *it );}

        associative_node_iterator& operator ++() { ++it; return *this;}
        associative_node_iterator operator ++( int ) { associative_node_iterator old( *this ); ++*this; return old;}
        associative_node_iterator& operator --() { --it; return *this;}
        associative_node_iterator operator --( int ) { associative_node_iterator old( *this ); --*this; return old;}


        bool operator == ( const associative_node_iterator<stored_type, tree_type, container_type, const tree_type*, const tree_type&>& rhs ) const { return associative_node_it_eq( this, rhs ); }
        bool operator != ( const associative_node_iterator<stored_type, tree_type, container_type, const tree_type*, const tree_type&>& rhs ) const { return !( *this == rhs ); }

    protected:

        typename container_type::const_iterator it;
        const associative_tree<stored_type, tree_type, container_type>* pParent;
        // 120 "child_node_iterator.h"
        template<typename T, typename U, typename V> friend class basic_tree;
        template<typename T, typename U, typename V> friend class associative_tree;
        template<typename T, typename U, typename V, typename W, typename X, typename Y> friend class pre_order_descendant_node_iterator;
        template<typename T, typename U, typename V, typename W, typename X, typename Y> friend class post_order_descendant_node_iterator;
        friend void associative_node_it_init<>( associative_node_iterator<stored_type, tree_type, container_type, const tree_type*, const tree_type&>*, const associative_node_iterator<stored_type, tree_type, container_type, tree_type*, tree_type&>& );
        friend void associative_node_it_init<>( associative_node_iterator<stored_type, tree_type, container_type, tree_type*, tree_type&>*, const associative_node_iterator<stored_type, tree_type, container_type, tree_type*, tree_type&>& );
        friend bool associative_node_it_eq<>( const associative_node_iterator<stored_type, tree_type, container_type, const tree_type*, const tree_type&>*, const associative_node_iterator<stored_type, tree_type, container_type, const tree_type*, const tree_type&>& );
        friend bool associative_node_it_eq<>( const associative_node_iterator<stored_type, tree_type, container_type, tree_type*, tree_type&>*, const associative_node_iterator<stored_type, tree_type, container_type, const tree_type*, const tree_type&>& );

};
// 138 "child_node_iterator.h"
template<typename stored_type, typename tree_type, typename container_type, typename pointer_type, typename reference_type>
class tcl::sequential_node_iterator



    : public std::iterator<std::bidirectional_iterator_tag, tree_type, ptrdiff_t, pointer_type, reference_type>

{
    public:

        sequential_node_iterator() : pParent( 0 ) {}

        sequential_node_iterator( const sequential_node_iterator<stored_type, tree_type, container_type, tree_type*, tree_type&>& src ) { sequential_node_it_init( this, src ); }
    protected:



        explicit sequential_node_iterator( typename container_type::const_iterator it_, const tree_type* pParent_ ) : it( it_ ), pParent( pParent_ ) {}


    public:

        typedef size_t size_type;



        typedef typename std::iterator_traits<sequential_node_iterator>::difference_type difference_type;



        reference_type operator*() const { return const_cast<reference_type>( *( *it ) );}
        pointer_type operator->() const { return const_cast<pointer_type>( *it );}

        sequential_node_iterator& operator ++() { ++it; return *this;}
        sequential_node_iterator operator ++( int ) { sequential_node_iterator old( *this ); ++*this; return old;}
        sequential_node_iterator& operator --() { --it; return *this;}
        sequential_node_iterator operator --( int ) { sequential_node_iterator old( *this ); --*this; return old;}
        sequential_node_iterator& operator +=( size_type n ) { it += n; return *this;}
        sequential_node_iterator& operator -=( size_type n ) { it -= n; return *this;}
        difference_type operator -( const sequential_node_iterator& rhs ) const { return it - rhs.it;}


        bool operator == ( const sequential_node_iterator<stored_type, tree_type, container_type, const tree_type*, const tree_type&>& rhs ) const { return sequential_node_it_eq( this, rhs ); }
        bool operator != ( const sequential_node_iterator<stored_type, tree_type, container_type, const tree_type*, const tree_type&>& rhs ) const { return !( *this == rhs ); }
        bool operator < ( const sequential_node_iterator<stored_type, tree_type, container_type, const tree_type*, const tree_type&>& rhs ) const { return sequential_node_it_less( this, rhs ); }
        bool operator <= ( const sequential_node_iterator<stored_type, tree_type, container_type, const tree_type*, const tree_type&>& rhs ) const { return *this < rhs || *this == rhs; }
        bool operator > ( const sequential_node_iterator<stored_type, tree_type, container_type, const tree_type*, const tree_type&>& rhs ) const { return !( *this <= rhs ); }
        bool operator >= ( const sequential_node_iterator<stored_type, tree_type, container_type, const tree_type*, const tree_type&>& rhs ) const { return !( *this < rhs ); }

    protected:

        typename container_type::const_iterator it;
        const tree_type* pParent;
        // 202 "child_node_iterator.h"
        template<typename T> friend class sequential_tree;
        template<typename T, typename U, typename V, typename W, typename X, typename Y> friend class pre_order_descendant_node_iterator;
        template<typename T, typename U, typename V, typename W, typename X, typename Y> friend class post_order_descendant_node_iterator;
        friend void sequential_node_it_init<>( sequential_node_iterator<stored_type, tree_type, container_type, const tree_type*, const tree_type&>*, const sequential_node_iterator<stored_type, tree_type, container_type, tree_type*, tree_type&>& );
        friend void sequential_node_it_init<>( sequential_node_iterator<stored_type, tree_type, container_type, tree_type*, tree_type&>*, const sequential_node_iterator<stored_type, tree_type, container_type, tree_type*, tree_type&>& );
        friend bool sequential_node_it_eq<>( const sequential_node_iterator<stored_type, tree_type, container_type, const tree_type*, const tree_type&>*, const sequential_node_iterator<stored_type, tree_type, container_type, const tree_type*, const tree_type&>& );
        friend bool sequential_node_it_eq<>( const sequential_node_iterator<stored_type, tree_type, container_type, tree_type*, tree_type&>*, const sequential_node_iterator<stored_type, tree_type, container_type, const tree_type*, const tree_type&>& );
        friend bool sequential_node_it_less<>( const sequential_node_iterator<stored_type, tree_type, container_type, const tree_type*, const tree_type&>*, const sequential_node_iterator<stored_type, tree_type, container_type, const tree_type*, const tree_type&>& );
        friend bool sequential_node_it_less<>( const sequential_node_iterator<stored_type, tree_type, container_type, tree_type*, tree_type&>*, const sequential_node_iterator<stored_type, tree_type, container_type, const tree_type*, const tree_type&>& );

};
// 31 "associative_tree.h" 2
// 1 "descendant_iterator.h" 1
// 27 "descendant_iterator.h"







namespace tcl
{
    template<typename T, typename U, typename V, typename W, typename X, typename Y, typename Z> class pre_order_descendant_iterator;
    template<typename T, typename U, typename V, typename W, typename X, typename Y, typename Z> class post_order_descendant_iterator;
    template<typename T, typename U, typename V, typename W, typename X, typename Y, typename Z> class level_order_descendant_iterator;
    template<typename T, typename U> class tree;
    template<typename T, typename U> class multitree;
    template<typename T, typename U, typename V> class unique_tree;

    template<typename ST, typename TT, typename TPT1, typename CT, typename BIT1, typename PT1, typename RT1, typename TPT2, typename BIT2, typename PT2, typename RT2>
    void pre_order_it_init( pre_order_descendant_iterator<ST, TT, TPT1, CT, BIT1, PT1, RT1>* dest, const pre_order_descendant_iterator<ST, TT, TPT2, CT, BIT2, PT2, RT2>& src )
    {
        dest->it = src.it;
        dest->pTop_node = src.pTop_node;
        dest->at_top = src.at_top;
        std::stack<typename TT::iterator> s( src.node_stack );
        std::stack<typename TT::iterator> temp;

        while ( !s.empty() ) { temp.push( s.top() ); s.pop(); }

        while ( !temp.empty() ) { dest->node_stack.push( temp.top() ); temp.pop(); }
    }

    template<typename ST, typename TT, typename TPT1, typename CT, typename BIT1, typename PT1, typename RT1, typename TPT2, typename BIT2, typename PT2, typename RT2>
    bool pre_order_it_eq( const pre_order_descendant_iterator<ST, TT, TPT1, CT, BIT1, PT1, RT1>* lhs, const pre_order_descendant_iterator<ST, TT, TPT2, CT, BIT2, PT2, RT2>& rhs ) { return lhs->it == rhs.it && lhs->at_top == rhs.at_top; }

    template<typename ST, typename TT, typename TPT1, typename CT, typename BIT1, typename PT1, typename RT1, typename TPT2, typename BIT2, typename PT2, typename RT2>
    void post_order_it_init( post_order_descendant_iterator<ST, TT, TPT1, CT, BIT1, PT1, RT1>* dest, const post_order_descendant_iterator<ST, TT, TPT2, CT, BIT2, PT2, RT2>& src )
    {
        dest->it = src.it;
        dest->pTop_node = src.pTop_node;
        dest->at_top = src.at_top;
        std::stack<typename TT::iterator> s( src.node_stack );
        std::stack<typename TT::iterator> temp;

        while ( !s.empty() ) { temp.push( s.top() ); s.pop(); }

        while ( !temp.empty() ) { dest->node_stack.push( temp.top() ); temp.pop(); }
    }

    template<typename ST, typename TT, typename TPT1, typename CT, typename BIT1, typename PT1, typename RT1, typename TPT2, typename BIT2, typename PT2, typename RT2>
    bool post_order_it_eq( const post_order_descendant_iterator<ST, TT, TPT1, CT, BIT1, PT1, RT1>* lhs, const post_order_descendant_iterator<ST, TT, TPT2, CT, BIT2, PT2, RT2>& rhs ) { return lhs->it == rhs.it && lhs->at_top == rhs.at_top; }

    template<typename ST, typename TT, typename TPT1, typename CT, typename BIT1, typename PT1, typename RT1, typename TPT2, typename BIT2, typename PT2, typename RT2>
    void level_order_it_init( level_order_descendant_iterator<ST, TT, TPT1, CT, BIT1, PT1, RT1>* dest, const level_order_descendant_iterator<ST, TT, TPT2, CT, BIT2, PT2, RT2>& src )
    {
        dest->it = src.it;
        dest->pTop_node = src.pTop_node;
        dest->at_top = src.at_top;
        std::queue<typename TT::iterator> temp( src.node_queue );

        while ( !temp.empty() ) {dest->node_queue.push( temp.front() ); temp.pop();}
    }

    template<typename ST, typename TT, typename TPT1, typename CT, typename BIT1, typename PT1, typename RT1, typename TPT2, typename BIT2, typename PT2, typename RT2>
    bool level_order_it_eq( const level_order_descendant_iterator<ST, TT, TPT1, CT, BIT1, PT1, RT1>* lhs, const level_order_descendant_iterator<ST, TT, TPT2, CT, BIT2, PT2, RT2>& rhs ) { return lhs->it == rhs.it && lhs->at_top == rhs.at_top; }
}






template<typename stored_type, typename tree_type, typename tree_pointer_type, typename container_type, typename base_iterator_type, typename pointer_type, typename reference_type>
class tcl::pre_order_descendant_iterator



    : public std::iterator<std::bidirectional_iterator_tag, stored_type, ptrdiff_t, pointer_type, reference_type>

{
    public:

        pre_order_descendant_iterator() : pTop_node( 0 ), at_top( false ) {}

        pre_order_descendant_iterator( const pre_order_descendant_iterator<stored_type, tree_type, tree_type*, container_type, typename tree_type::iterator, stored_type*, stored_type&>& src ) { pre_order_it_init( this, src ); }
        virtual ~pre_order_descendant_iterator() {}

    private:
        explicit pre_order_descendant_iterator( tree_pointer_type pCalled_node, bool beg ) : it( beg ? pCalled_node->begin() : pCalled_node->end() ), pTop_node( pCalled_node ), at_top( beg ) {}

    public:

        pre_order_descendant_iterator& operator ++();
        pre_order_descendant_iterator operator ++( int ) { pre_order_descendant_iterator old( *this ); ++*this; return old;}
        pre_order_descendant_iterator& operator --();
        pre_order_descendant_iterator operator --( int ) { pre_order_descendant_iterator old( *this ); --*this; return old;}


        reference_type operator*() const { return at_top ? *pTop_node->get() : it.operator * ();}
        pointer_type operator->() const { return at_top ? pTop_node->get() : it.operator ->();}
        tree_pointer_type node() const { return at_top ? pTop_node : it.node();}
        base_iterator_type base() const { return it; }



        bool operator == ( const pre_order_descendant_iterator<stored_type, tree_type, const tree_type*, container_type, typename tree_type::const_iterator, const stored_type*, const stored_type&>& rhs ) const { return pre_order_it_eq( this, rhs ); }
        bool operator != ( const pre_order_descendant_iterator<stored_type, tree_type, const tree_type*, container_type, typename tree_type::const_iterator, const stored_type*, const stored_type&>& rhs ) const { return !( *this == rhs ); }

    private:

        base_iterator_type it;
        std::stack<base_iterator_type> node_stack;
        tree_pointer_type pTop_node;
        typename container_type::const_reverse_iterator rit;
        bool at_top;
        // 146 "descendant_iterator.h"
        template<typename T, typename U> friend class tree;
        template<typename T, typename U> friend class multitree;
        template<typename T, typename U, typename V> friend class unique_tree;
        template<typename T> friend class sequential_tree;
        friend void pre_order_it_init<>( pre_order_descendant_iterator<stored_type, tree_type, const tree_type*, container_type, typename tree_type::const_iterator, const stored_type*, const stored_type&>*, const pre_order_descendant_iterator<stored_type, tree_type, tree_type*, container_type, typename tree_type::iterator, stored_type*, stored_type&>& );
        friend void pre_order_it_init<>( pre_order_descendant_iterator<stored_type, tree_type, tree_type*, container_type, typename tree_type::iterator, stored_type*, stored_type&>*, const pre_order_descendant_iterator<stored_type, tree_type, tree_type*, container_type, typename tree_type::iterator, stored_type*, stored_type&>& );
        friend bool pre_order_it_eq<>( const pre_order_descendant_iterator<stored_type, tree_type, const tree_type*, container_type, typename tree_type::const_iterator, const stored_type*, const stored_type&>*, const pre_order_descendant_iterator<stored_type, tree_type, const tree_type*, container_type, typename tree_type::const_iterator, const stored_type*, const stored_type&>& );
        friend bool pre_order_it_eq<>( const pre_order_descendant_iterator<stored_type, tree_type, tree_type*, container_type, typename tree_type::iterator, stored_type*, stored_type&>*, const pre_order_descendant_iterator<stored_type, tree_type, const tree_type*, container_type, typename tree_type::const_iterator, const stored_type*, const stored_type&>& );

};






template<typename stored_type, typename tree_type, typename tree_pointer_type, typename container_type, typename base_iterator_type, typename pointer_type, typename reference_type>
class tcl::post_order_descendant_iterator



    : public std::iterator<std::bidirectional_iterator_tag, stored_type, ptrdiff_t, pointer_type, reference_type>

{
    public:

        post_order_descendant_iterator() : pTop_node( 0 ) {}
        virtual ~post_order_descendant_iterator() {}

        post_order_descendant_iterator( const post_order_descendant_iterator<stored_type, tree_type, tree_type*, container_type, typename tree_type::iterator, stored_type*, stored_type&>& src ) { post_order_it_init( this, src ); }

    private:
        explicit post_order_descendant_iterator( tree_pointer_type pCalled_node, bool beg );

    public:

        post_order_descendant_iterator& operator ++();
        post_order_descendant_iterator operator ++( int ) { post_order_descendant_iterator old( *this ); ++*this; return old;}
        post_order_descendant_iterator& operator --();
        post_order_descendant_iterator operator --( int ) { post_order_descendant_iterator old( *this ); --*this; return old;}


        reference_type operator*() const { return at_top ? *pTop_node->get() : it.operator * ();}
        pointer_type operator->() const { return at_top ? pTop_node->get() : it.operator ->();}
        tree_pointer_type node() const { return at_top ? pTop_node : it.node();}
        base_iterator_type base() const { return it; }


        bool operator == ( const post_order_descendant_iterator<stored_type, tree_type, const tree_type*, container_type, typename tree_type::const_iterator, const stored_type*, const stored_type&>& rhs ) const { return post_order_it_eq( this, rhs ); }
        bool operator != ( const post_order_descendant_iterator<stored_type, tree_type, const tree_type*, container_type, typename tree_type::const_iterator, const stored_type*, const stored_type&>& rhs ) const { return !( *this == rhs ); }

    private:

        std::stack<base_iterator_type> node_stack;
        base_iterator_type it;
        tree_pointer_type pTop_node;
        typename container_type::const_reverse_iterator rit;
        bool at_top;
        // 215 "descendant_iterator.h"
        template<typename T> friend class sequential_tree;
        template<typename T, typename U> friend class tree;
        template<typename T, typename U> friend class multitree;
        template<typename T, typename U, typename V> friend class unique_tree;
        friend void post_order_it_init<>( post_order_descendant_iterator<stored_type, tree_type, const tree_type*, container_type, typename tree_type::const_iterator, const stored_type*, const stored_type&>*, const post_order_descendant_iterator<stored_type, tree_type, tree_type*, container_type, typename tree_type::iterator, stored_type*, stored_type&>& );
        friend void post_order_it_init<>( post_order_descendant_iterator<stored_type, tree_type, tree_type*, container_type, typename tree_type::iterator, stored_type*, stored_type&>*, const post_order_descendant_iterator<stored_type, tree_type, tree_type*, container_type, typename tree_type::iterator, stored_type*, stored_type&>& );
        friend bool post_order_it_eq<>( const post_order_descendant_iterator<stored_type, tree_type, const tree_type*, container_type, typename tree_type::const_iterator, const stored_type*, const stored_type&>*, const post_order_descendant_iterator<stored_type, tree_type, const tree_type*, container_type, typename tree_type::const_iterator, const stored_type*, const stored_type&>& );
        friend bool post_order_it_eq<>( const post_order_descendant_iterator<stored_type, tree_type, tree_type*, container_type, typename tree_type::iterator, stored_type*, stored_type&>*, const post_order_descendant_iterator<stored_type, tree_type, const tree_type*, container_type, typename tree_type::const_iterator, const stored_type*, const stored_type&>& );

};






template<typename stored_type, typename tree_type, typename tree_pointer_type, typename container_type, typename base_iterator_type, typename pointer_type, typename reference_type>
class tcl::level_order_descendant_iterator



    : public std::iterator<std::forward_iterator_tag, stored_type, ptrdiff_t, pointer_type, reference_type>

{
    public:

        level_order_descendant_iterator() : pTop_node( 0 ), at_top( false ) {}

        level_order_descendant_iterator( const level_order_descendant_iterator<stored_type, tree_type, tree_type*, container_type, typename tree_type::iterator, stored_type*, stored_type&>& src ) { level_order_it_init( this, src ); }
        virtual ~level_order_descendant_iterator() {}

    protected:
        explicit level_order_descendant_iterator( tree_pointer_type pCalled_node, bool beg ) : it( beg ? pCalled_node->begin() : pCalled_node->end() ), pTop_node( pCalled_node ), at_top( beg ) {}

    public:

        level_order_descendant_iterator& operator ++();
        level_order_descendant_iterator operator ++( int ) { level_order_descendant_iterator old( *this ); ++*this; return old;}


        reference_type operator*() const { return at_top ? *pTop_node->get() : it.operator * ();}
        pointer_type operator->() const { return at_top ? pTop_node->get() : it.operator ->();}
        tree_pointer_type node() const { return at_top ? pTop_node : it.node();}
        base_iterator_type base() const { return it; }


        bool operator == ( const level_order_descendant_iterator<stored_type, tree_type, const tree_type*, container_type, typename tree_type::const_iterator, const stored_type*, const stored_type&>& rhs ) const { return level_order_it_eq( this, rhs ); }
        bool operator != ( const level_order_descendant_iterator<stored_type, tree_type, const tree_type*, container_type, typename tree_type::const_iterator, const stored_type*, const stored_type&>& rhs ) const { return !( *this == rhs ); }

    private:

        base_iterator_type it;
        std::queue<base_iterator_type> node_queue;
        tree_pointer_type pTop_node;
        bool at_top;
        // 281 "descendant_iterator.h"
        template<typename T, typename U> friend class tree;
        template<typename T, typename U> friend class multitree;
        template<typename T, typename U, typename V> friend class unique_tree;
        template<typename T> friend class sequential_tree;
        friend void level_order_it_init<>( level_order_descendant_iterator<stored_type, tree_type, const tree_type*, container_type, typename tree_type::const_iterator, const stored_type*, const stored_type&>*, const level_order_descendant_iterator<stored_type, tree_type, tree_type*, container_type, typename tree_type::iterator, stored_type*, stored_type&>& );
        friend void level_order_it_init<>( level_order_descendant_iterator<stored_type, tree_type, tree_type*, container_type, typename tree_type::iterator, stored_type*, stored_type&>*, const level_order_descendant_iterator<stored_type, tree_type, tree_type*, container_type, typename tree_type::iterator, stored_type*, stored_type&>& );
        friend bool level_order_it_eq<>( const level_order_descendant_iterator<stored_type, tree_type, const tree_type*, container_type, typename tree_type::const_iterator, const stored_type*, const stored_type&>*, const level_order_descendant_iterator<stored_type, tree_type, const tree_type*, container_type, typename tree_type::const_iterator, const stored_type*, const stored_type&>& );
        friend bool level_order_it_eq<>( const level_order_descendant_iterator<stored_type, tree_type, tree_type*, container_type, typename tree_type::iterator, stored_type*, stored_type&>*, const level_order_descendant_iterator<stored_type, tree_type, const tree_type*, container_type, typename tree_type::const_iterator, const stored_type*, const stored_type&>& );

};


// 1 "descendant_iterator.inl" 1
// 29 "descendant_iterator.inl"
template<typename stored_type, typename tree_type, typename tree_pointer_type, typename container_type, typename base_iterator_type, typename pointer_type, typename reference_type>
tcl::pre_order_descendant_iterator<stored_type, tree_type, tree_pointer_type, container_type, base_iterator_type, pointer_type, reference_type>& tcl::pre_order_descendant_iterator<stored_type, tree_type, tree_pointer_type, container_type, base_iterator_type, pointer_type, reference_type>::operator ++()
{
    if ( at_top )
    {
        at_top = false;
        it = pTop_node->begin();
    }

    else if ( !it.node()->empty() )
    {
        node_stack.push( it );
        it = it.node()->begin();
    }

    else
    {
        ++it;

        while ( !node_stack.empty() && it == ( node_stack.top() ).node()->end() )
        {
            it = node_stack.top();
            node_stack.pop();
            ++it;
        }
    }

    return *this;
}


template<typename stored_type, typename tree_type, typename tree_pointer_type, typename container_type, typename base_iterator_type, typename pointer_type, typename reference_type>
tcl::pre_order_descendant_iterator<stored_type, tree_type, tree_pointer_type, container_type, base_iterator_type, pointer_type, reference_type>& tcl::pre_order_descendant_iterator<stored_type, tree_type, tree_pointer_type, container_type, base_iterator_type, pointer_type, reference_type>::operator --()
{
    if ( it == pTop_node->end() )
    {
        if ( pTop_node->empty() )
        {
            at_top = true;
            return *this;
        }

        rit = pTop_node->children.rbegin();

        if ( rit != const_cast<const tree_type*>( pTop_node )->children.rend() )
        {
            if ( !( *rit )->empty() )
            {
                do
                {
                    ++rit;
                    it = base_iterator_type( rit.base(), ( it != pTop_node->end() ? it.node() : pTop_node ) );
                    node_stack.push( it );
                    rit = it.node()->children.rbegin();
                }
                while ( !( *rit )->empty() );
            }

            ++rit;
            it = base_iterator_type( rit.base(), ( it != pTop_node->end() ? node() : pTop_node ) );
        }
    }

    else
    {
        if ( it != it.node()->parent()->begin() )
        {
            --it;

            if ( !it.node()->empty() )
            {
                do
                {
                    node_stack.push( it );
                    it = base_iterator_type( it.node()->children.end(), it.node() );
                    --it;
                }
                while ( !it.node()->empty() );
            }
        }

        else if ( !node_stack.empty() )
        {
            it = node_stack.top();
            node_stack.pop();
        }

        else
        {
            if ( !at_top )
            {
                at_top = true;
            }

            else
            {
                --it;
                at_top = false;
            }
        }
    }

    return *this;
}


template<typename stored_type, typename tree_type, typename tree_pointer_type, typename container_type, typename base_iterator_type, typename pointer_type, typename reference_type>
tcl::post_order_descendant_iterator<stored_type, tree_type, tree_pointer_type, container_type, base_iterator_type, pointer_type, reference_type>::post_order_descendant_iterator( tree_pointer_type pCalled_node, bool beg ) : pTop_node( pCalled_node ), at_top( false )
{
    if ( !beg )
    {
        it = pTop_node->end();
    }

    else
    {
        it = pCalled_node->begin();

        if ( it != pTop_node->end() )
        {
            if ( !it.node()->empty() )
            {
                do
                {
                    node_stack.push( it );
                    it = it.node()->begin();
                }
                while ( !it.node()->empty() );
            }
        }

        else
        {
            at_top = true;
        }
    }
}


template<typename stored_type, typename tree_type, typename tree_pointer_type, typename container_type, typename base_iterator_type, typename pointer_type, typename reference_type>
tcl::post_order_descendant_iterator<stored_type, tree_type, tree_pointer_type, container_type, base_iterator_type, pointer_type, reference_type>& tcl::post_order_descendant_iterator<stored_type, tree_type, tree_pointer_type, container_type, base_iterator_type, pointer_type, reference_type>::operator ++()
{
    if ( at_top )
    {
        at_top = false;
        it = pTop_node->end();
        return *this;
    }

    else if ( pTop_node->empty() )
    {
        ++it;
        return *this;
    }

    const base_iterator_type it_end = it.node()->parent()->end();
    ++it;

    if ( it != it_end && !it.node()->empty() )
    {
        do
        {
            node_stack.push( it );
            it = it.node()->begin();
        }
        while ( !it.node()->empty() );
    }

    else
    {
        if ( !node_stack.empty() && it == ( node_stack.top() ).node()->end() )
        {
            it = node_stack.top();
            node_stack.pop();
        }

        else if ( node_stack.empty() && it == pTop_node->end() )
        {
            at_top = true;
        }
    }

    return *this;
}


template<typename stored_type, typename tree_type, typename tree_pointer_type, typename container_type, typename base_iterator_type, typename pointer_type, typename reference_type>
tcl::post_order_descendant_iterator<stored_type, tree_type, tree_pointer_type, container_type, base_iterator_type, pointer_type, reference_type>& tcl::post_order_descendant_iterator<stored_type, tree_type, tree_pointer_type, container_type, base_iterator_type, pointer_type, reference_type>::operator --()
{
    if ( at_top )
    {
        at_top = false;
        typename container_type::const_reverse_iterator rit = pTop_node->children.rbegin();
        ++rit;
        it = base_iterator_type( rit.base(), pTop_node );
    }

    else if ( it == pTop_node->end() )
    {
        at_top = true;
    }

    else
    {
        if ( !it.node()->empty() )
        {
            typename container_type::const_reverse_iterator rit = it.node()->children.rbegin();
            node_stack.push( it );
            ++rit;
            it = base_iterator_type( rit.base(), it.node() );
        }

        else
        {
            if ( it != it.node()->parent()->begin() )
            {
                --it;
            }

            else
            {
                while ( !node_stack.empty() && it == node_stack.top().node()->begin() )
                {
                    it = node_stack.top();
                    node_stack.pop();
                }

                --it;
            }
        }
    }

    return *this;
}


template<typename stored_type, typename tree_type, typename tree_pointer_type, typename container_type, typename base_iterator_type, typename pointer_type, typename reference_type>
tcl::level_order_descendant_iterator<stored_type, tree_type, tree_pointer_type, container_type, base_iterator_type, pointer_type, reference_type>& tcl::level_order_descendant_iterator<stored_type, tree_type, tree_pointer_type, container_type, base_iterator_type, pointer_type, reference_type>::operator ++()
{
    if ( at_top )
    {
        at_top = false;
        it = pTop_node->begin();
        return *this;
    }

    const base_iterator_type it_end = it.node()->parent()->end();
    node_queue.push( it );
    ++it;

    if ( it == it_end )
    {
        while ( !node_queue.empty() )
        {
            it = node_queue.front();
            node_queue.pop();

            if ( !it.node()->empty() )
            {
                it = node()->begin();
                break;
            }

            else if ( node_queue.empty() )
            {
                it = pTop_node->end();
                return *this;
            }
        }
    }

    return *this;
}
// 294 "descendant_iterator.h" 2
// 32 "associative_tree.h" 2
// 1 "descendant_node_iterator.h" 1
// 27 "descendant_node_iterator.h"







namespace tcl
{
    template<typename T, typename U, typename V, typename W, typename X, typename Y> class pre_order_descendant_node_iterator;
    template<typename T, typename U, typename V, typename W, typename X, typename Y> class post_order_descendant_node_iterator;
    template<typename T, typename U, typename V, typename W, typename X, typename Y> class level_order_descendant_node_iterator;

    template<typename ST, typename TT, typename CT, typename BIT1, typename PT1, typename RT1, typename BIT2, typename PT2, typename RT2>
    void pre_order_node_it_init( pre_order_descendant_node_iterator<ST, TT, CT, BIT1, PT1, RT1>* dest, const pre_order_descendant_node_iterator<ST, TT, CT, BIT2, PT2, RT2>& src )
    {
        dest->it = src.it;
        dest->pTop_node = src.pTop_node;
        dest->at_top = src.at_top;
        std::stack<typename TT::node_iterator> s( src.node_stack );
        std::stack<typename TT::node_iterator> temp;

        while ( !s.empty() ) { temp.push( s.top() ); s.pop(); }

        while ( !temp.empty() ) { dest->node_stack.push( temp.top() ); temp.pop(); }
    }

    template<typename ST, typename TT, typename CT, typename BIT1, typename PT1, typename RT1, typename BIT2, typename PT2, typename RT2>
    bool pre_order_node_it_eq( const pre_order_descendant_node_iterator<ST, TT, CT, BIT1, PT1, RT1>* lhs, const pre_order_descendant_node_iterator<ST, TT, CT, BIT2, PT2, RT2>& rhs ) { return lhs->it == rhs.it && lhs->at_top == rhs.at_top; }

    template<typename ST, typename TT, typename CT, typename BIT1, typename PT1, typename RT1, typename BIT2, typename PT2, typename RT2>
    void post_order_node_it_init( post_order_descendant_node_iterator<ST, TT, CT, BIT1, PT1, RT1>* dest, const post_order_descendant_node_iterator<ST, TT, CT, BIT2, PT2, RT2>& src )
    {
        dest->it = src.it;
        dest->pTop_node = src.pTop_node;
        dest->at_top = src.at_top;
        std::stack<typename TT::node_iterator> s( src.node_stack );
        std::stack<typename TT::node_iterator> temp;

        while ( !s.empty() ) { temp.push( s.top() ); s.pop(); }

        while ( !temp.empty() ) { dest->node_stack.push( temp.top() ); temp.pop(); }
    }

    template<typename ST, typename TT, typename CT, typename BIT1, typename PT1, typename RT1, typename BIT2, typename PT2, typename RT2>
    bool post_order_node_it_eq( const post_order_descendant_node_iterator<ST, TT, CT, BIT1, PT1, RT1>* lhs, const post_order_descendant_node_iterator<ST, TT, CT, BIT2, PT2, RT2>& rhs ) { return lhs->it == rhs.it && lhs->at_top == rhs.at_top; }

    template<typename ST, typename TT, typename CT, typename BIT1, typename PT1, typename RT1, typename BIT2, typename PT2, typename RT2>
    void level_order_node_it_init( level_order_descendant_node_iterator<ST, TT, CT, BIT1, PT1, RT1>* dest, const level_order_descendant_node_iterator<ST, TT, CT, BIT2, PT2, RT2>& src )
    {
        dest->it = src.it;
        dest->pTop_node = src.pTop_node;
        dest->at_top = src.at_top;
        std::queue<typename TT::node_iterator> temp( src.node_queue );

        while ( !temp.empty() ) {dest->node_queue.push( temp.front() ); temp.pop();}
    }

    template<typename ST, typename TT, typename CT, typename BIT1, typename PT1, typename RT1, typename BIT2, typename PT2, typename RT2>
    bool level_order_node_it_eq( const level_order_descendant_node_iterator<ST, TT, CT, BIT1, PT1, RT1>* lhs, const level_order_descendant_node_iterator<ST, TT, CT, BIT2, PT2, RT2>& rhs ) { return lhs->it == rhs.it && lhs->at_top == rhs.at_top; }

}





template<typename stored_type, typename tree_type, typename container_type, typename base_iterator_type, typename pointer_type, typename reference_type>
class tcl::pre_order_descendant_node_iterator



    : public std::iterator<std::bidirectional_iterator_tag, tree_type, ptrdiff_t, pointer_type, reference_type>

{
    public:

        pre_order_descendant_node_iterator() : pTop_node( 0 ), at_top( false ) {}

        pre_order_descendant_node_iterator( const pre_order_descendant_node_iterator<stored_type, tree_type, container_type, typename tree_type::node_iterator, tree_type*, tree_type&>& src ) { pre_order_node_it_init( this, src ); }

    protected:
        explicit pre_order_descendant_node_iterator( pointer_type pCalled_node, bool beg ) : it( beg ? pCalled_node->node_begin() : pCalled_node->node_end() ), pTop_node( pCalled_node ), at_top( beg ) {}

    public:

        pre_order_descendant_node_iterator& operator ++();
        pre_order_descendant_node_iterator operator ++( int ) { pre_order_descendant_node_iterator old( *this ); ++*this; return old;}
        pre_order_descendant_node_iterator& operator --();
        pre_order_descendant_node_iterator operator --( int ) { pre_order_descendant_node_iterator old( *this ); --*this; return old;}

        reference_type operator*() const { return at_top ? *pTop_node : it.operator * ();}
        pointer_type operator->() const { return at_top ? pTop_node : it.operator ->();}
        base_iterator_type base() const { return it; }


        bool operator == ( const pre_order_descendant_node_iterator<stored_type, tree_type, container_type, typename tree_type::const_node_iterator, const tree_type*, const tree_type&>& rhs ) const { return pre_order_node_it_eq( this, rhs ); }
        bool operator != ( const pre_order_descendant_node_iterator<stored_type, tree_type, container_type, typename tree_type::const_node_iterator, const tree_type*, const tree_type&>& rhs ) const { return !( *this == rhs ); }

    private:

        base_iterator_type it;
        std::stack<base_iterator_type> node_stack;
        pointer_type pTop_node;
        typename container_type::const_reverse_iterator rit;
        bool at_top;
        // 139 "descendant_node_iterator.h"
        template<typename T, typename U> friend class tree;
        template<typename T, typename U> friend class multitree;
        template<typename T, typename U, typename V> friend class unique_tree;
        template<typename T> friend class sequential_tree;
        friend void pre_order_node_it_init<>( pre_order_descendant_node_iterator<stored_type, tree_type, container_type, typename tree_type::const_node_iterator, const tree_type*, const tree_type&>*, const pre_order_descendant_node_iterator<stored_type, tree_type, container_type, typename tree_type::node_iterator, tree_type*, tree_type&>& );
        friend void pre_order_node_it_init<>( pre_order_descendant_node_iterator<stored_type, tree_type, container_type, typename tree_type::node_iterator, tree_type*, tree_type&>*, const pre_order_descendant_node_iterator<stored_type, tree_type, container_type, typename tree_type::node_iterator, tree_type*, tree_type&>& );
        friend bool pre_order_node_it_eq<>( const pre_order_descendant_node_iterator<stored_type, tree_type, container_type, typename tree_type::const_node_iterator, const tree_type*, const tree_type&>*, const pre_order_descendant_node_iterator<stored_type, tree_type, container_type, typename tree_type::const_node_iterator, const tree_type*, const tree_type&>& );
        friend bool pre_order_node_it_eq<>( const pre_order_descendant_node_iterator<stored_type, tree_type, container_type, typename tree_type::node_iterator, tree_type*, tree_type&>*, const pre_order_descendant_node_iterator<stored_type, tree_type, container_type, typename tree_type::const_node_iterator, const tree_type*, const tree_type&>& );

};






template<typename stored_type, typename tree_type, typename container_type, typename base_iterator_type, typename pointer_type, typename reference_type>
class tcl::post_order_descendant_node_iterator



    : public std::iterator<std::bidirectional_iterator_tag, tree_type, ptrdiff_t, pointer_type, reference_type>

{
    public:

        post_order_descendant_node_iterator() : pTop_node( 0 ), at_top( false ) {}

        post_order_descendant_node_iterator( const post_order_descendant_node_iterator<stored_type, tree_type, container_type, typename tree_type::node_iterator, tree_type*, tree_type&>& src ) { post_order_node_it_init( this, src ); }

    private:
        explicit post_order_descendant_node_iterator( pointer_type pCalled_node, bool beg );

    public:

        post_order_descendant_node_iterator& operator ++();
        post_order_descendant_node_iterator operator ++( int ) { post_order_descendant_node_iterator old( *this ); ++*this; return old;}
        post_order_descendant_node_iterator& operator --();
        post_order_descendant_node_iterator operator --( int ) { post_order_descendant_node_iterator old( *this ); --*this; return old;}


        reference_type operator*() const { return at_top ? *pTop_node : it.operator * ();}
        pointer_type operator->() const { return at_top ? pTop_node : it.operator ->();}
        base_iterator_type base() const { return it; }


        bool operator == ( const post_order_descendant_node_iterator<stored_type, tree_type, container_type, typename tree_type::const_node_iterator, const tree_type*, const tree_type&>& rhs ) const { return post_order_node_it_eq( this, rhs ); }
        bool operator != ( const post_order_descendant_node_iterator<stored_type, tree_type, container_type, typename tree_type::const_node_iterator, const tree_type*, const tree_type&>& rhs ) const { return !( *this == rhs ); }

    private:

        std::stack<base_iterator_type> node_stack;
        base_iterator_type it;
        pointer_type pTop_node;
        typename container_type::const_reverse_iterator rit;
        bool at_top;
        // 206 "descendant_node_iterator.h"
        template<typename T, typename U> friend class tree;
        template<typename T, typename U> friend class multitree;
        template<typename T, typename U, typename V> friend class unique_tree;
        template<typename T> friend class sequential_tree;
        friend void post_order_node_it_init<>( post_order_descendant_node_iterator<stored_type, tree_type, container_type, typename tree_type::const_node_iterator, const tree_type*, const tree_type&>*, const post_order_descendant_node_iterator<stored_type, tree_type, container_type, typename tree_type::node_iterator, tree_type*, tree_type&>& );
        friend void post_order_node_it_init<>( post_order_descendant_node_iterator<stored_type, tree_type, container_type, typename tree_type::node_iterator, tree_type*, tree_type&>*, const post_order_descendant_node_iterator<stored_type, tree_type, container_type, typename tree_type::node_iterator, tree_type*, tree_type&>& );
        friend bool post_order_node_it_eq<>( const post_order_descendant_node_iterator<stored_type, tree_type, container_type, typename tree_type::const_node_iterator, const tree_type*, const tree_type&>*, const post_order_descendant_node_iterator<stored_type, tree_type, container_type, typename tree_type::const_node_iterator, const tree_type*, const tree_type&>& );
        friend bool post_order_node_it_eq<>( const post_order_descendant_node_iterator<stored_type, tree_type, container_type, typename tree_type::node_iterator, tree_type*, tree_type&>*, const post_order_descendant_node_iterator<stored_type, tree_type, container_type, typename tree_type::const_node_iterator, const tree_type*, const tree_type&>& );

};







template<typename stored_type, typename tree_type, typename container_type, typename base_iterator_type, typename pointer_type, typename reference_type>
class tcl::level_order_descendant_node_iterator



    : public std::iterator<std::forward_iterator_tag, tree_type, ptrdiff_t, pointer_type, reference_type>

{
    public:

        level_order_descendant_node_iterator() : pTop_node( 0 ), at_top( false ) {}

        level_order_descendant_node_iterator( const level_order_descendant_node_iterator<stored_type, tree_type, container_type, typename tree_type::node_iterator, tree_type*, tree_type&>& src ) { level_order_node_it_init( this, src ); }

    private:
        explicit level_order_descendant_node_iterator( pointer_type pCalled_node, bool beg ) : it( beg ? pCalled_node->node_begin() : pCalled_node->node_end() ), pTop_node( pCalled_node ), at_top( beg ) {}

    public:

        level_order_descendant_node_iterator& operator ++();
        level_order_descendant_node_iterator operator ++( int ) { level_order_descendant_node_iterator old( *this ); ++*this; return old;}


        reference_type operator*() const { return at_top ? const_cast<reference_type>( *pTop_node ) : const_cast<reference_type>( it.operator * () );}
        pointer_type operator->() const { return at_top ? const_cast<pointer_type>( pTop_node ) : const_cast<pointer_type>( it.operator ->() );}
        base_iterator_type base() const { return it; }


        bool operator == ( const level_order_descendant_node_iterator<stored_type, tree_type, container_type, typename tree_type::const_node_iterator, const tree_type*, const tree_type&>& rhs ) const { return level_order_node_it_eq( this, rhs ); }
        bool operator != ( const level_order_descendant_node_iterator<stored_type, tree_type, container_type, typename tree_type::const_node_iterator, const tree_type*, const tree_type&>& rhs ) const { return !( *this == rhs ); }

    private:

        base_iterator_type it;
        std::queue<base_iterator_type> node_queue;
        pointer_type pTop_node;
        bool at_top;
        // 271 "descendant_node_iterator.h"
        template<typename T, typename U> friend class tree;
        template<typename T, typename U> friend class multitree;
        template<typename T, typename U, typename V> friend class unique_tree;
        template<typename T> friend class sequential_tree;
        friend void level_order_node_it_init<>( level_order_descendant_node_iterator<stored_type, tree_type, container_type, typename tree_type::const_node_iterator, const tree_type*, const tree_type&>*, const level_order_descendant_node_iterator<stored_type, tree_type, container_type, typename tree_type::node_iterator, tree_type*, tree_type&>& );
        friend void level_order_node_it_init<>( level_order_descendant_node_iterator<stored_type, tree_type, container_type, typename tree_type::node_iterator, tree_type*, tree_type&>*, const level_order_descendant_node_iterator<stored_type, tree_type, container_type, typename tree_type::node_iterator, tree_type*, tree_type&>& );
        friend bool level_order_node_it_eq<>( const level_order_descendant_node_iterator<stored_type, tree_type, container_type, typename tree_type::const_node_iterator, const tree_type*, const tree_type&>*, const level_order_descendant_node_iterator<stored_type, tree_type, container_type, typename tree_type::const_node_iterator, const tree_type*, const tree_type&>& );
        friend bool level_order_node_it_eq<>( const level_order_descendant_node_iterator<stored_type, tree_type, container_type, typename tree_type::node_iterator, tree_type*, tree_type&>*, const level_order_descendant_node_iterator<stored_type, tree_type, container_type, typename tree_type::const_node_iterator, const tree_type*, const tree_type&>& );

};



// 1 "descendant_node_iterator.inl" 1
// 29 "descendant_node_iterator.inl"
template<typename stored_type, typename tree_type, typename container_type, typename base_iterator_type, typename pointer_type, typename reference_type>
tcl::pre_order_descendant_node_iterator<stored_type, tree_type, container_type, base_iterator_type, pointer_type, reference_type>& tcl::pre_order_descendant_node_iterator<stored_type, tree_type, container_type, base_iterator_type, pointer_type, reference_type>::operator ++()
{
    if ( at_top )
    {
        at_top = false;
        it = pTop_node->node_begin();
    }

    else if ( !it->empty() )
    {
        node_stack.push( it );
        it = it->node_begin();
    }

    else
    {
        ++it;

        while ( !node_stack.empty() && it == ( node_stack.top() )->node_end() )
        {
            it = node_stack.top();
            node_stack.pop();
            ++it;
        }
    }

    return *this;
}


template<typename stored_type, typename tree_type, typename container_type, typename base_iterator_type, typename pointer_type, typename reference_type>
tcl::pre_order_descendant_node_iterator<stored_type, tree_type, container_type, base_iterator_type, pointer_type, reference_type>& tcl::pre_order_descendant_node_iterator<stored_type, tree_type, container_type, base_iterator_type, pointer_type, reference_type>::operator --()
{
    if ( it == pTop_node->node_end() )
    {
        if ( pTop_node->empty() )
        {
            at_top = true;
            return *this;
        }

        rit = pTop_node->children.rbegin();

        if ( rit != const_cast<const tree_type*>( pTop_node )->children.rend() )
        {
            if ( !( *rit )->empty() )
            {
                do
                {
                    ++rit;
                    it = base_iterator_type( rit.base(), ( it != pTop_node->node_end() ? & ( *it ) : pTop_node ) );
                    node_stack.push( it );
                    rit = it->children.rbegin();
                }
                while ( !( *rit )->empty() );
            }

            ++rit;
            it = base_iterator_type( rit.base(), ( it != pTop_node->node_end() ? & ( *it ) : pTop_node ) );
        }
    }

    else
    {
        if ( it != it->parent()->node_begin() )
        {
            --it;

            if ( !it->empty() )
            {
                do
                {
                    node_stack.push( it );
                    it = base_iterator_type( it->children.end(), &( *it ) );
                    --it;
                }
                while ( !it->empty() );
            }
        }

        else if ( !node_stack.empty() )
        {
            it = node_stack.top();
            node_stack.pop();
        }

        else
        {
            if ( !at_top )
            {
                at_top = true;
            }

            else
            {
                --it;
                at_top = false;
            }
        }
    }

    return *this;
}


template<typename stored_type, typename tree_type, typename container_type, typename base_iterator_type, typename pointer_type, typename reference_type>
tcl::post_order_descendant_node_iterator<stored_type, tree_type, container_type, base_iterator_type, pointer_type, reference_type>::post_order_descendant_node_iterator( pointer_type pCalled_node, bool beg ) : pTop_node( pCalled_node ), at_top( false )
{
    if ( !beg )
    {
        it = pTop_node->node_end();
    }

    else
    {
        it = pTop_node->node_begin();

        if ( it != pTop_node->node_end() )
        {
            if ( !it->empty() )
            {
                do
                {
                    node_stack.push( it );
                    it = it->node_begin();
                }
                while ( !it->empty() );
            }
        }

        else
        {
            at_top = true;
        }
    }
}


template<typename stored_type, typename tree_type, typename container_type, typename base_iterator_type, typename pointer_type, typename reference_type>
tcl::post_order_descendant_node_iterator<stored_type, tree_type, container_type, base_iterator_type, pointer_type, reference_type>& tcl::post_order_descendant_node_iterator<stored_type, tree_type, container_type, base_iterator_type, pointer_type, reference_type>::operator ++()
{
    if ( at_top )
    {
        at_top = false;
        it = pTop_node->node_end();
        return *this;
    }

    else if ( pTop_node->empty() )
    {
        ++it;
        return *this;
    }

    const base_iterator_type it_end = it->parent()->node_end();
    ++it;

    if ( it != it_end && !it->empty() )
    {
        do
        {
            node_stack.push( it );
            it = it->node_begin();
        }
        while ( !it->empty() );
    }

    else
    {
        if ( !node_stack.empty() && it == node_stack.top()->node_end() )
        {
            it = node_stack.top();
            node_stack.pop();
        }

        else if ( node_stack.empty() && it == pTop_node->node_end() )
        {
            at_top = true;
        }
    }

    return *this;
}


template<typename stored_type, typename tree_type, typename container_type, typename base_iterator_type, typename pointer_type, typename reference_type>
tcl::post_order_descendant_node_iterator<stored_type, tree_type, container_type, base_iterator_type, pointer_type, reference_type>& tcl::post_order_descendant_node_iterator<stored_type, tree_type, container_type, base_iterator_type, pointer_type, reference_type>::operator --()
{
    if ( at_top )
    {
        at_top = false;
        typename container_type::const_reverse_iterator rit = pTop_node->children.rbegin();
        ++rit;
        it = base_iterator_type( rit.base(), pTop_node );
    }

    else if ( it == pTop_node->node_end() )
    {
        at_top = true;
    }

    else
    {
        if ( !it->empty() )
        {
            typename container_type::const_reverse_iterator rit = it->children.rbegin();
            node_stack.push( it );
            ++rit;
            it = base_iterator_type( rit.base(), &( *it ) );
        }

        else
        {
            if ( it != it->parent()->node_begin() )
            {
                --it;
            }

            else
            {
                while ( !node_stack.empty() && it == node_stack.top()->node_begin() )
                {
                    it = node_stack.top();
                    node_stack.pop();
                }

                --it;
            }
        }
    }

    return *this;
}


template<typename stored_type, typename tree_type, typename container_type, typename base_iterator_type, typename pointer_type, typename reference_type>
tcl::level_order_descendant_node_iterator<stored_type, tree_type, container_type, base_iterator_type, pointer_type, reference_type>& tcl::level_order_descendant_node_iterator<stored_type, tree_type, container_type, base_iterator_type, pointer_type, reference_type>::operator ++()
{
    if ( at_top )
    {
        at_top = false;
        it = pTop_node->node_begin();
        return *this;
    }

    const base_iterator_type it_end = it->parent()->node_end();
    node_queue.push( it );
    ++it;

    if ( it == it_end )
    {
        while ( !node_queue.empty() )
        {
            it = node_queue.front();
            node_queue.pop();

            if ( !it->empty() )
            {
                it = it->node_begin();
                break;
            }

            else if ( node_queue.empty() )
            {
                it = pTop_node->node_end();
                return *this;
            }
        }
    }

    return *this;
}
// 285 "descendant_node_iterator.h" 2
// 33 "associative_tree.h" 2
// 1 "reverse_iterator.h" 1
// 27 "reverse_iterator.h"




namespace tcl
{
    template<typename T, typename U, typename V, typename W, typename X, typename Y> class associative_reverse_iterator;
    template<typename T, typename U, typename V, typename W, typename X, typename Y> class const_sequential_reverse_iterator;
}



template<typename stored_type, typename tree_type, typename tree_pointer_type, typename container_type, typename pointer_type, typename reference_type>
class tcl::associative_reverse_iterator : public tcl::associative_iterator<stored_type, tree_type, tree_pointer_type, container_type, pointer_type, reference_type>
{
        typedef associative_iterator<stored_type, tree_type, tree_pointer_type, container_type, pointer_type, reference_type> associative_iterator_type;
    public:
        associative_reverse_iterator() : associative_iterator_type() {}
        explicit associative_reverse_iterator( const associative_iterator_type& _it ) : associative_iterator_type( _it ) {}

        associative_reverse_iterator( const associative_reverse_iterator<stored_type, tree_type, tree_type*, container_type, stored_type*, stored_type&>& src ) : associative_iterator_type( src ) {}

        reference_type operator*() const { associative_iterator_type tmp( *this ); return( *--tmp );}
        pointer_type operator->() const { associative_iterator_type tmp( *this ); --tmp; return tmp.operator ->();}
        associative_reverse_iterator& operator ++() { associative_iterator_type::operator --(); return *this;}
        associative_reverse_iterator operator ++( int ) { associative_reverse_iterator old( *this ); ++*this; return old;}
        associative_reverse_iterator& operator --() { associative_iterator_type::operator ++(); return *this;}
        associative_reverse_iterator operator --( int ) { associative_reverse_iterator old( *this ); --*this; return old;}

        tree_pointer_type node() const { associative_iterator_type tmp( *this ); --tmp; return tmp.node();}
        associative_iterator_type base() const { return associative_iterator_type( *this );}
};




template<typename stored_type, typename tree_type, typename tree_pointer_type, typename container_type, typename pointer_type, typename reference_type>
class tcl::sequential_reverse_iterator : public tcl::sequential_iterator<stored_type, tree_type, tree_pointer_type, container_type, pointer_type, reference_type>
{
        typedef sequential_iterator<stored_type, tree_type, tree_pointer_type, container_type, pointer_type, reference_type> sequential_iterator_type;
    public:
        sequential_reverse_iterator() : sequential_iterator_type() {}
        explicit sequential_reverse_iterator( const sequential_iterator_type& _it ) : sequential_iterator_type( _it ) {}

        sequential_reverse_iterator( const sequential_reverse_iterator<stored_type, tree_type, tree_type*, container_type, stored_type*, stored_type&>& src ) : sequential_iterator_type( src ) {}

        reference_type operator*() const { sequential_iterator_type tmp( *this ); return( *--tmp );}
        pointer_type operator->() const { sequential_iterator_type tmp( *this ); --tmp; return tmp.operator ->();}
        sequential_reverse_iterator& operator ++() { sequential_iterator_type::operator --(); return *this;}
        sequential_reverse_iterator operator ++( int ) { sequential_reverse_iterator old( *this ); ++*this; return old;}
        sequential_reverse_iterator& operator --() { sequential_iterator_type::operator ++(); return *this;}
        sequential_reverse_iterator operator --( int ) { sequential_reverse_iterator old( *this ); --*this; return old;}

        tree_pointer_type node() const { sequential_iterator_type tmp( *this ); --tmp; return tmp.node();}
        sequential_iterator_type base() const { return sequential_iterator_type( *this );}
};
// 34 "associative_tree.h" 2
// 1 "reverse_node_iterator.h" 1
// 27 "reverse_node_iterator.h"




namespace tcl
{
    template<typename T, typename U, typename V, typename W, typename X> class associative_reverse_node_iterator;
    template<typename T, typename U, typename V, typename W, typename X> class sequential_reverse_node_iterator;
}


template<typename stored_type, typename tree_type, typename container_type, typename pointer_type, typename reference_type>
class tcl::associative_reverse_node_iterator : public tcl::associative_node_iterator<stored_type, tree_type, container_type, pointer_type, reference_type>
{
        typedef associative_node_iterator<stored_type, tree_type, container_type, pointer_type, reference_type> associative_iterator_type;
    public:
        associative_reverse_node_iterator() : associative_iterator_type() {}
        explicit associative_reverse_node_iterator( const associative_iterator_type& _it ) : associative_iterator_type( _it ) {}

        reference_type operator*() const { associative_iterator_type tmp( *this ); return( *--tmp );}
        pointer_type operator->() const { associative_iterator_type tmp( *this ); --tmp; return tmp.operator ->();}
        associative_reverse_node_iterator& operator ++() { associative_iterator_type::operator --(); return *this;}
        associative_reverse_node_iterator operator ++( int ) { associative_reverse_node_iterator old( *this ); ++*this; return old;}
        associative_reverse_node_iterator& operator --() { associative_iterator_type::operator ++(); return *this;}
        associative_reverse_node_iterator operator --( int ) { associative_reverse_node_iterator old( *this ); --*this; return old;}

        associative_iterator_type base() const { return associative_iterator_type( *this );}
};




template<typename stored_type, typename tree_type, typename container_type, typename pointer_type, typename reference_type>
class tcl::sequential_reverse_node_iterator : public tcl::sequential_node_iterator<stored_type, tree_type, container_type, pointer_type, reference_type>
{
        typedef sequential_node_iterator<stored_type, tree_type, container_type, pointer_type, reference_type> sequential_iterator_type;
    public:
        sequential_reverse_node_iterator() : sequential_iterator_type() {}
        explicit sequential_reverse_node_iterator( const sequential_iterator_type& _it ) : sequential_iterator_type( _it ) {}

        reference_type operator*() const { sequential_iterator_type tmp( *this ); return( *--tmp );}
        pointer_type operator->() const { sequential_iterator_type tmp( *this ); --tmp; return tmp.operator ->();}
        sequential_reverse_node_iterator& operator ++() { sequential_iterator_type::operator --(); return *this;}
        sequential_reverse_node_iterator operator ++( int ) { sequential_reverse_node_iterator old( *this ); ++*this; return old;}
        sequential_reverse_node_iterator& operator --() { sequential_iterator_type::operator ++(); return *this;}
        sequential_reverse_node_iterator operator --( int ) { sequential_reverse_node_iterator old( *this ); --*this; return old;}

        sequential_iterator_type base() const { return sequential_iterator_type( *this );}
};
// 35 "associative_tree.h" 2


namespace tcl
{
    template<typename T, typename U, typename X> class associative_tree;


    template<typename T, typename U, typename V> bool operator == ( const associative_tree<T, U, V>& lhs, const associative_tree<T, U, V>& rhs );
    template<typename T, typename U, typename V> bool operator < ( const associative_tree<T, U, V>& lhs, const associative_tree<T, U, V>& rhs );
    template<typename T, typename U, typename V> bool operator != ( const associative_tree<T, U, V>& lhs, const associative_tree<T, U, V>& rhs ) { return !( lhs == rhs );}
    template<typename T, typename U, typename V> bool operator > ( const associative_tree<T, U, V>& lhs, const associative_tree<T, U, V>& rhs ) { return rhs < lhs;}
    template<typename T, typename U, typename V> bool operator <= ( const associative_tree<T, U, V>& lhs, const associative_tree<T, U, V>& rhs ) { return !( rhs < lhs );}
    template<typename T, typename U, typename V> bool operator >= ( const associative_tree<T, U, V>& lhs, const associative_tree<T, U, V>& rhs ) { return !( lhs < rhs );}
}





template< typename stored_type, typename tree_type, typename container_type >
class tcl::associative_tree : public basic_tree<stored_type, tree_type, container_type>
{
    protected:
        typedef basic_tree<stored_type, tree_type, container_type> basic_tree_type;
        explicit associative_tree( const stored_type& stored_obj ) : basic_tree_type( stored_obj ) {}
        virtual ~associative_tree() {}

    public:

        typedef associative_tree<stored_type, tree_type, container_type> associative_tree_type;
        typedef stored_type key_type;
        using basic_tree_type::size_type;


        typedef associative_iterator<stored_type, tree_type, const tree_type*, container_type, const stored_type*, const stored_type&> const_iterator;
        typedef associative_iterator<stored_type, tree_type, tree_type*, container_type, stored_type*, stored_type&> iterator;
        typedef associative_reverse_iterator<stored_type, tree_type, const tree_type*, container_type, const stored_type*, const stored_type&> const_reverse_iterator;
        typedef associative_reverse_iterator<stored_type, tree_type, tree_type*, container_type, stored_type*, stored_type&> reverse_iterator;
        typedef pre_order_descendant_iterator<stored_type, tree_type, const tree_type*, container_type, const_iterator, const stored_type*, const stored_type&> const_pre_order_iterator;
        typedef pre_order_descendant_iterator<stored_type, tree_type, tree_type*, container_type, iterator, stored_type*, stored_type&> pre_order_iterator;
        typedef post_order_descendant_iterator<stored_type, tree_type, const tree_type*, container_type, const_iterator, const stored_type*, const stored_type&> const_post_order_iterator;
        typedef post_order_descendant_iterator<stored_type, tree_type, tree_type*, container_type, iterator, stored_type*, stored_type&> post_order_iterator;
        typedef level_order_descendant_iterator<stored_type, tree_type, const tree_type*, container_type, const_iterator, const stored_type*, const stored_type&> const_level_order_iterator;
        typedef level_order_descendant_iterator<stored_type, tree_type, tree_type*, container_type, iterator, stored_type*, stored_type&> level_order_iterator;


        typedef associative_node_iterator<stored_type, tree_type, container_type, const tree_type*, const tree_type&> const_node_iterator;
        typedef associative_node_iterator<stored_type, tree_type, container_type, tree_type*, tree_type&> node_iterator;
        typedef associative_reverse_node_iterator<stored_type, tree_type, container_type, const tree_type*, const tree_type&> const_reverse_node_iterator;
        typedef associative_reverse_node_iterator<stored_type, tree_type, container_type, tree_type*, tree_type&> reverse_node_iterator;
        typedef pre_order_descendant_node_iterator<stored_type, tree_type, container_type, const_node_iterator, const tree_type*, const tree_type&> const_pre_order_node_iterator;
        typedef pre_order_descendant_node_iterator<stored_type, tree_type, container_type, node_iterator, tree_type*, tree_type&> pre_order_node_iterator;
        typedef post_order_descendant_node_iterator<stored_type, tree_type, container_type, const_node_iterator, const tree_type*, const tree_type&> const_post_order_node_iterator;
        typedef post_order_descendant_node_iterator<stored_type, tree_type, container_type, node_iterator, tree_type*, tree_type&> post_order_node_iterator;
        typedef level_order_descendant_node_iterator<stored_type, tree_type, container_type, const_node_iterator, const tree_type*, const tree_type&> const_level_order_node_iterator;
        typedef level_order_descendant_node_iterator<stored_type, tree_type, container_type, node_iterator, tree_type*, tree_type&> level_order_node_iterator;


        const_iterator begin() const { return const_iterator( basic_tree_type::children.begin(), this );}
        const_iterator end() const { return const_iterator( basic_tree_type::children.end(), this );}
        iterator begin() { return iterator( basic_tree_type::children.begin(), this );}
        iterator end() { return iterator( basic_tree_type::children.end(), this );}


        const_node_iterator node_begin() const { return const_node_iterator( basic_tree_type::children.begin(), this );}
        const_node_iterator node_end() const { return const_node_iterator( basic_tree_type::children.end(), this );}
        node_iterator node_begin() { return node_iterator( basic_tree_type::children.begin(), this );}
        node_iterator node_end() { return node_iterator( basic_tree_type::children.end(), this );}


        const_reverse_iterator rbegin() const {return const_reverse_iterator( end() );}
        const_reverse_iterator rend() const {return const_reverse_iterator( begin() );}
        reverse_iterator rbegin() {return reverse_iterator( end() );}
        reverse_iterator rend() { return reverse_iterator( begin() );}


        const_reverse_node_iterator node_rbegin() const {return const_reverse_node_iterator( node_end() );}
        const_reverse_node_iterator node_rend() const {return const_reverse_node_iterator( node_begin() );}
        reverse_node_iterator node_rbegin() {return reverse_node_iterator( node_end() );}
        reverse_node_iterator node_rend() { return reverse_node_iterator( node_begin() );}


        iterator find( const stored_type& value );
        const_iterator find( const stored_type& value ) const;
        bool erase( const stored_type& value );
        void erase( iterator it );
        void erase( iterator it_beg, iterator it_end );
        void clear();
        typename basic_tree_type::size_type count( const stored_type& value ) const;
        iterator lower_bound( const stored_type& value );
        const_iterator lower_bound( const stored_type& value ) const;
        iterator upper_bound( const stored_type& value );
        const_iterator upper_bound( const stored_type& value ) const;
        std::pair<iterator, iterator> equal_range( const stored_type& value )
        {
            tree_type node_obj( value );
            iterator lower_it( basic_tree_type::children.lower_bound( &node_obj ), this );
            iterator upper_it( basic_tree_type::children.upper_bound( &node_obj ), this );
            return std::make_pair( lower_it, upper_it );
        }
        std::pair<const_iterator, const_iterator> equal_range( const stored_type& value ) const
        {
            tree_type node_obj( value );
            const_iterator lower_it( basic_tree_type::children.lower_bound( &node_obj ), this );
            const_iterator upper_it( basic_tree_type::children.upper_bound( &node_obj ), this );
            return std::make_pair( lower_it, upper_it );
        }

    protected:
        iterator insert( const stored_type& value, tree_type* parent ) { return insert( end(), value, parent );}
        iterator insert( const const_iterator& pos, const stored_type& value, tree_type* parent );
        iterator insert( const tree_type& tree_obj, tree_type* parent ) { return insert( end(), tree_obj, parent );}
        iterator insert( const const_iterator pos, const tree_type& tree_obj, tree_type* parent );
        void set( const tree_type& tree_obj, tree_type* parent );
        // 178 "associative_tree.h"
        template<typename T, typename U, typename V, typename W, typename X, typename Y, typename Z> friend class pre_order_descendant_iterator;
        template<typename T, typename U, typename V, typename W, typename X, typename Y, typename Z> friend class post_order_descendant_iterator;
        template<typename T, typename U, typename V, typename W, typename X, typename Y, typename Z> friend class level_order_descendant_iterator;
        template<typename T, typename U, typename V, typename W, typename X, typename Y> friend class pre_order_descendant_node_iterator;
        template<typename T, typename U, typename V, typename W, typename X, typename Y> friend class post_order_descendant_node_iterator;
        template<typename T, typename U, typename V, typename W, typename X, typename Y> friend class level_order_descendant_node_iterator;
        friend bool operator ==<> ( const associative_tree_type& lhs, const associative_tree_type& rhs );
        friend bool operator < <> ( const associative_tree_type& lhs, const associative_tree_type& rhs );

};

// 1 "associative_tree.inl" 1
// 29 "associative_tree.inl"
template< typename stored_type, typename tree_type, typename container_type >
typename tcl::associative_tree<stored_type, tree_type, container_type>::iterator
tcl::associative_tree<stored_type, tree_type, container_type>::find( const stored_type& value )
{
    tree_type node_obj( value );
    iterator it( basic_tree_type::children.find( &node_obj ), this );
    return it;
}


template< typename stored_type, typename tree_type, typename container_type >
typename tcl::associative_tree<stored_type, tree_type, container_type>::const_iterator
tcl::associative_tree<stored_type, tree_type, container_type>::find( const stored_type& value ) const
{
    tree_type node_obj( value );
    const_iterator it( basic_tree_type::children.find( &node_obj ), this );
    return it;
}



template< typename stored_type, typename tree_type, typename container_type >
bool tcl::associative_tree<stored_type, tree_type, container_type>::
erase( const stored_type& value )
{
    bool erased_nodes = false;
    tree_type node_obj( value );
    typename container_type::iterator it = basic_tree_type::children.find( &node_obj );

    while ( it != basic_tree_type::children.end() )
    {
        deallocate_tree_type( *it );
        basic_tree_type::children.erase( it );
        it = basic_tree_type::children.find( &node_obj );
        erased_nodes = true;
    }

    return erased_nodes;
}


template< typename stored_type, typename tree_type, typename container_type >
typename tcl::basic_tree<stored_type, tree_type, container_type>::size_type
tcl::associative_tree<stored_type, tree_type, container_type>::count( const stored_type& value ) const
{
    const_iterator it = find( value );
    const_iterator it_end = end();
    typename basic_tree_type::size_type cnt = 0;

    while ( it != it_end && !( *it < value || value < *it ) )
    {
        ++cnt;
        ++it;
    }

    return cnt;
}


template< typename stored_type, typename tree_type, typename container_type>
typename tcl::associative_tree<stored_type, tree_type, container_type>::iterator
tcl::associative_tree<stored_type, tree_type, container_type>::lower_bound( const stored_type& value )
{
    tree_type node_obj( value );
    iterator it( basic_tree_type::children.lower_bound( &node_obj ), this );
    return it;
}


template< typename stored_type, typename tree_type, typename container_type>
typename tcl::associative_tree<stored_type, tree_type, container_type>::const_iterator
tcl::associative_tree<stored_type, tree_type, container_type>::lower_bound( const stored_type& value ) const
{
    tree_type node_obj( value );
    const_iterator it( basic_tree_type::children.lower_bound( &node_obj ), this );
    return it;
}


template< typename stored_type, typename tree_type, typename container_type>
typename tcl::associative_tree<stored_type, tree_type, container_type>::iterator
tcl::associative_tree<stored_type, tree_type, container_type>::upper_bound( const stored_type& value )
{
    tree_type node_obj( value );
    iterator it( basic_tree_type::children.upper_bound( &node_obj ), this );
    return it;
}


template< typename stored_type, typename tree_type, typename container_type>
typename tcl::associative_tree<stored_type, tree_type, container_type>::const_iterator
tcl::associative_tree<stored_type, tree_type, container_type>::upper_bound( const stored_type& value ) const
{
    tree_type node_obj( value );
    const_iterator it( basic_tree_type::children.upper_bound( &node_obj ), this );
    return it;
}


template< typename stored_type, typename tree_type, typename container_type>
typename tcl::associative_tree<stored_type, tree_type, container_type>::iterator
tcl::associative_tree<stored_type, tree_type, container_type>::insert( const const_iterator& pos, const stored_type& value, tree_type* pParent )
{
    tree_type* pNew_node;
    basic_tree_type::allocate_tree_type( pNew_node, tree_type( value ) );
    pNew_node->set_parent( pParent );
    const typename basic_tree_type::size_type sz = basic_tree_type::children.size();
    typename container_type::iterator children_pos;

    if ( pos == pParent->end() )
    {
        children_pos = basic_tree_type::children.end();
    }

    else
    {
        children_pos = basic_tree_type::children.begin();
        typename container_type::const_iterator const_children_pos = basic_tree_type::children.begin();

        while ( const_children_pos != pos.it && const_children_pos != basic_tree_type::children.end() )
        {
            ++children_pos;
            ++const_children_pos;
        }
    }

    const typename container_type::iterator it = basic_tree_type::children.insert( children_pos, pNew_node );

    if ( sz == basic_tree_type::children.size() )
    {
        basic_tree_type::deallocate_tree_type( pNew_node );
        return iterator( basic_tree_type::children.end(), this );
    }

    return iterator( it, this );
}


template< typename stored_type, typename tree_type, typename container_type>
typename tcl::associative_tree<stored_type, tree_type, container_type>::iterator
tcl::associative_tree<stored_type, tree_type, container_type>::insert( const const_iterator pos, const tree_type& tree_obj, tree_type* pParent )
{
    iterator base_it = pParent->insert( pos, *tree_obj.get() );

    if ( base_it != pParent->end() )
    {
        const_iterator it = tree_obj.begin();
        const const_iterator it_end = tree_obj.end();

        for ( ; it != it_end; ++it )
        { base_it.node()->insert( *it.node() ); }
    }

    return base_it;
}


template< typename stored_type, typename tree_type, typename container_type>
void tcl::associative_tree<stored_type, tree_type, container_type>::set( const tree_type& tree_obj, tree_type* pParent )
{
    set( *tree_obj.get() );
    const_iterator it = tree_obj.begin();
    const const_iterator it_end = tree_obj.end();

    for ( ; it != it_end; ++it )
    {
        insert( *it.node(), pParent );
    }
}


template< typename stored_type, typename tree_type, typename container_type>
void tcl::associative_tree<stored_type, tree_type, container_type>::clear()
{
    iterator it = begin();
    const iterator it_end = end();

    for ( ; it != it_end; ++it )
    {
        basic_tree_type::deallocate_tree_type( it.node() );
    }

    basic_tree_type::children.clear();
}


template< typename stored_type, typename tree_type, typename container_type>
void tcl::associative_tree<stored_type, tree_type, container_type>::erase( iterator it )
{
    if ( it.pParent != this )
    { return; }

    it.node()->clear();
    deallocate_tree_type( it.node() );
    const iterator beg_it = begin();
    typename container_type::iterator pos_it = basic_tree_type::children.begin();

    for ( ; it != beg_it; --it, ++pos_it ) ;

    basic_tree_type::children.erase( pos_it );
}


template< typename stored_type, typename tree_type, typename container_type>
void tcl::associative_tree<stored_type, tree_type, container_type>::erase( iterator it_beg, iterator it_end )
{
    while ( it_beg != it_end )
    {
        erase( it_beg++ );
    }
}



template<typename stored_type, typename tree_type, typename container_type>
bool tcl::operator == ( const associative_tree<stored_type, tree_type, container_type>& lhs, const associative_tree<stored_type, tree_type, container_type>& rhs )
{
    if ( ( *lhs.get() < *rhs.get() ) || ( *rhs.get() < *lhs.get() ) )
    { return false; }

    typename associative_tree<stored_type, tree_type, container_type>::const_iterator lhs_it = lhs.begin();
    const typename associative_tree<stored_type, tree_type, container_type>::const_iterator lhs_end = lhs.end();
    typename associative_tree<stored_type, tree_type, container_type>::const_iterator rhs_it = rhs.begin();
    const typename associative_tree<stored_type, tree_type, container_type>::const_iterator rhs_end = rhs.end();

    for ( ; lhs_it != lhs_end && rhs_it != rhs_end; ++lhs_it, ++rhs_it )
    {
        if ( *lhs_it.node() != *rhs_it.node() )
        {
            return false;
        }
    }

    if ( lhs_it != lhs.end() || rhs_it != rhs.end() )
    { return false; }

    return true;
}



template<typename stored_type, typename tree_type, typename container_type>
bool tcl::operator < ( const associative_tree<stored_type, tree_type, container_type>& lhs, const associative_tree<stored_type, tree_type, container_type>& rhs )
{
    if ( *lhs.get() < *rhs.get() )
    { return true; }

    typename associative_tree<stored_type, tree_type, container_type>::const_iterator lhs_it = lhs.begin();
    const typename associative_tree<stored_type, tree_type, container_type>::const_iterator lhs_end = lhs.end();
    typename associative_tree<stored_type, tree_type, container_type>::const_iterator rhs_it = rhs.begin();
    const typename associative_tree<stored_type, tree_type, container_type>::const_iterator rhs_end = rhs.end();

    for ( ; lhs_it != lhs_end && rhs_it != rhs_end; ++lhs_it, ++rhs_it )
    {
        if ( *lhs_it.node() < *rhs_it.node() )
        {
            return true;
        }
    }

    if ( lhs.size() != rhs.size() )
    {
        return lhs.size() < rhs.size();
    }

    return false;
}
// 190 "associative_tree.h" 2
// 29 "Tree.h" 2


namespace tcl
{

    template<typename stored_type, typename node_compare_type > class tree;


    template<typename stored_type, typename node_compare_type >
    struct tree_deref_less : public std::binary_function<const tree<stored_type, node_compare_type>*, const tree<stored_type, node_compare_type>*, bool>
    {
        bool operator()( const tree<stored_type, node_compare_type>* lhs, const tree<stored_type, node_compare_type>* rhs ) const
        {
            return node_compare_type()( *lhs->get(), *rhs->get() );
        }
    };
}






template<typename stored_type, typename node_compare_type = std::less<stored_type> >
class tcl::tree : public tcl::associative_tree<stored_type, tcl::tree<stored_type, node_compare_type>, std::set<tcl::tree<stored_type, node_compare_type>*, tcl::tree_deref_less<stored_type, node_compare_type> > >
{
    public:

        typedef tree<stored_type, node_compare_type> tree_type;
        typedef tree_deref_less<stored_type, node_compare_type> key_compare;
        typedef tree_deref_less<stored_type, node_compare_type> value_compare;
        typedef std::set<tree_type*, key_compare> container_type;
        typedef basic_tree<stored_type, tree_type, container_type> basic_tree_type;
        typedef associative_tree<stored_type, tree_type, container_type> associative_tree_type;


        explicit tree( const stored_type& value = stored_type() ) : associative_tree_type( value ) {}
        template<typename iterator_type> tree( iterator_type it_beg, iterator_type it_end, const stored_type& value = stored_type() ) : associative_tree_type( value )
        {
            while ( it_beg != it_end )
            {
                insert( *it_beg );
                ++it_beg;
            }
        }
        tree( const tree_type& rhs );
        ~tree() { associative_tree_type::clear();}


        tree_type& operator = ( const tree_type& rhs );


    public:
        typename associative_tree_type::iterator insert( const stored_type& value ) { return associative_tree_type::insert( value, this );}
        typename associative_tree_type::iterator insert( const typename associative_tree_type::const_iterator pos, const stored_type& value ) { return associative_tree_type::insert( pos, value, this );}
        typename associative_tree_type::iterator insert( const tree_type& tree_obj ) { return associative_tree_type::insert( tree_obj, this );}
        typename associative_tree_type::iterator insert( const typename associative_tree_type::const_iterator pos, const tree_type& tree_obj ) { return associative_tree_type::insert( pos, tree_obj, this );}

        template<typename iterator_type> void insert( iterator_type it_beg, iterator_type it_end ) { while ( it_beg != it_end ) { insert( *it_beg++ ); }}

        void swap( tree_type& rhs );


        typedef typename associative_tree_type::post_order_iterator post_order_iterator_type;
        typedef typename associative_tree_type::const_post_order_iterator const_post_order_iterator_type;
        typedef typename associative_tree_type::pre_order_iterator pre_order_iterator_type;
        typedef typename associative_tree_type::const_pre_order_iterator const_pre_order_iterator_type;
        typedef typename associative_tree_type::level_order_iterator level_order_iterator_type;
        typedef typename associative_tree_type::const_level_order_iterator const_level_order_iterator_type;

        pre_order_iterator_type pre_order_begin() { return pre_order_iterator_type( this, true );}
        pre_order_iterator_type pre_order_end() { return pre_order_iterator_type( this, false );}
        const_pre_order_iterator_type pre_order_begin() const { return const_pre_order_iterator_type( this, true );}
        const_pre_order_iterator_type pre_order_end() const { return const_pre_order_iterator_type( this, false );}
        post_order_iterator_type post_order_begin() { return post_order_iterator_type( this, true );}
        post_order_iterator_type post_order_end() { return post_order_iterator_type( this, false );}
        const_post_order_iterator_type post_order_begin() const { return const_post_order_iterator_type( this, true );}
        const_post_order_iterator_type post_order_end() const { return const_post_order_iterator_type( this, false );}
        level_order_iterator_type level_order_begin() { return level_order_iterator_type( this, true );}
        level_order_iterator_type level_order_end() { return level_order_iterator_type( this, false );}
        const_level_order_iterator_type level_order_begin() const { return const_level_order_iterator_type( this, true );}
        const_level_order_iterator_type level_order_end() const { return const_level_order_iterator_type( this, false );}


        typedef typename associative_tree_type::pre_order_node_iterator pre_order_node_iterator_type;
        typedef typename associative_tree_type::const_pre_order_node_iterator const_pre_order_node_iterator_type;
        typedef typename associative_tree_type::post_order_node_iterator post_order_node_iterator_type;
        typedef typename associative_tree_type::const_post_order_node_iterator const_post_order_node_iterator_type;
        typedef typename associative_tree_type::level_order_node_iterator level_order_node_iterator_type;
        typedef typename associative_tree_type::const_level_order_node_iterator const_level_order_node_iterator_type;

        pre_order_node_iterator_type pre_order_node_begin() { return pre_order_node_iterator_type( this, true );}
        pre_order_node_iterator_type pre_order_node_end() { return pre_order_node_iterator_type( this, false );}
        const_pre_order_node_iterator_type pre_order_node_begin() const { return const_pre_order_node_iterator_type( this, true );}
        const_pre_order_node_iterator_type pre_order_node_end() const { return const_pre_order_node_iterator_type( this, false );}
        post_order_node_iterator_type post_order_node_begin() { return post_order_node_iterator_type( this, true );}
        post_order_node_iterator_type post_order_node_end() { return post_order_node_iterator_type( this, false );}
        const_post_order_node_iterator_type post_order_node_begin() const { return const_post_order_node_iterator_type( this, true );}
        const_post_order_node_iterator_type post_order_node_end() const { return const_post_order_node_iterator_type( this, false );}
        level_order_node_iterator_type level_order_node_begin() { return level_order_node_iterator_type( this, true );}
        level_order_node_iterator_type level_order_node_end() { return level_order_node_iterator_type( this, false );}
        const_level_order_node_iterator_type level_order_node_begin() const { return const_level_order_node_iterator_type( this, true );}
        const_level_order_node_iterator_type level_order_node_end() const { return const_level_order_node_iterator_type( this, false );}





        template<typename T, typename U, typename V> friend class basic_tree;

};

// 1 "tree.inl" 1
// 29 "tree.inl"
template<typename stored_type, typename node_compare_type>
tcl::tree<stored_type, node_compare_type>::tree( const tree_type& rhs ) : associative_tree_type( rhs )
{
    typename associative_tree_type::const_iterator it = rhs.begin();
    const typename associative_tree_type::const_iterator it_end = rhs.end();

    for ( ; it != it_end; ++it )
    {
        associative_tree_type::insert( *it.node(), this );
    }
}


template<typename stored_type, typename node_compare_type>
tcl::tree<stored_type, node_compare_type>&
tcl::tree<stored_type, node_compare_type>::operator = ( const tree_type& rhs )
{
    if ( !associative_tree_type::is_root() )
    { return *this; }

    if ( this == &rhs )
    { return *this; }

    associative_tree_type::clear();
    basic_tree_type::operator =( rhs );
    typename associative_tree_type::const_iterator it = rhs.begin();
    const typename associative_tree_type::const_iterator it_end = rhs.end();

    for ( ; it != it_end; ++it )
    {
        associative_tree_type::insert( *it.node(), this );
    }

    return *this;
}


template<typename stored_type, typename node_compare_type>
void tcl::tree<stored_type, node_compare_type>::swap( tree_type& rhs )
{
    tree_type temp( *this );
    associative_tree_type::clear();
    *this = rhs;
    rhs.clear();
    rhs = temp;
}
// 138 "Tree.h" 2
