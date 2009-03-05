#pragma once

#include <tchar.h>

// TODO: simplify/refactor/improve

template <class Derived>
class StringFormaterBase
{
public:
  typedef std::basic_string<_TCHAR> string_t;
  typedef std::basic_ostringstream<_TCHAR> ostringstream_t;
  
  StringFormaterBase(string_t fmt) : m_fmt(fmt) {}

  template <class T>
  Derived & operator% (const T & param)
  {
    ostringstream_t os;
    os << param;
    m_params.push_back(os.str());
    return *(static_cast<Derived*>(this));
  }

private:
  string_t m_fmt;
  std::vector<string_t> m_params;
  string_t m_result;

protected:
  const string_t & DoFormat()
  {
    // TODO: correct unicode support
    ostringstream_t os;

    size_t pos = 0;
    while (pos < m_fmt.size())
    {
      if (m_fmt[pos] != '{')
      {
        os << m_fmt[pos++];
        continue;
      }

      // parse int in "{}"
      size_t p1 = pos + 1;
      unsigned int i = 0;
      while ( p1 < m_fmt.size() && isdigit(m_fmt[p1]) )
      {
        i = i * 10 + m_fmt[p1++] - '0';
        if (i >= m_params.size())
        {
          os << m_fmt[pos++];
          continue;
        }
      }

      if (p1 >= m_fmt.size() || m_fmt[p1] != '}')
      {
        os << m_fmt[pos++];
        continue;
      }

      os << m_params[i];
      pos = p1 + 1;
    }

    m_result = os.str();
    return m_result;
  }
};

class StringFormaterStr : public StringFormaterBase<StringFormaterStr>
{
public:
  StringFormaterStr(string_t fmt) : StringFormaterBase<StringFormaterStr>(fmt) {}

  operator string_t()
  {
    return DoFormat();
  }
};

class StringFormaterCStr : public StringFormaterBase<StringFormaterCStr>
{
public:
  StringFormaterCStr(string_t fmt) : StringFormaterBase<StringFormaterCStr>(fmt) {}

  operator const _TCHAR * ()
  {
    return DoFormat().c_str();
  }
};

inline StringFormaterCStr format(StringFormaterCStr::string_t fmt)
{
  return StringFormaterCStr(fmt);
}

inline StringFormaterStr formatStr(StringFormaterStr::string_t fmt)
{
  return StringFormaterStr(fmt);
}

inline StringFormaterCStr formatCStr(StringFormaterCStr::string_t fmt)
{
  return StringFormaterCStr(fmt);
}
