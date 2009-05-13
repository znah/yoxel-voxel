#pragma once

class noncopyable
{
 protected:
    noncopyable() {}
    ~noncopyable() {}
 private:  // emphasize the following members are private
    noncopyable( const noncopyable& );
    const noncopyable& operator=( const noncopyable& );
};

template <class Stream, class T>
inline void write(Stream & s, const T & data)
{
  s.write( reinterpret_cast<const char *>(&data), sizeof(data) );
}

template <class Stream, class T>
inline void read(Stream & s, T & data)
{
  s.read( reinterpret_cast<char *>(&data), sizeof(data) );
}
