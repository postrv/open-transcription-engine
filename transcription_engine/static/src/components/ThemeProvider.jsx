import React, { createContext, useContext, useEffect, useState } from "react"

const ThemeProviderContext = createContext()

export function ThemeProvider({
  children,
  defaultTheme = "light",  // Explicitly set default to light
  storageKey = "vite-ui-theme",
  ...props
}) {
  const [theme, setTheme] = useState(() => {
    // Clear any stored theme first to ensure we start fresh
    if (typeof window !== 'undefined') {
      localStorage.removeItem(storageKey);
    }
    return defaultTheme;
  });

  useEffect(() => {
    const root = window.document.documentElement;
    root.classList.remove("light", "dark");

    // Ensure we add the light class
    root.classList.add(theme);

    // Store the theme
    if (typeof window !== 'undefined') {
      localStorage.setItem(storageKey, theme);
    }
  }, [theme, storageKey]);

  const value = {
    theme,
    setTheme: (newTheme) => {
      localStorage.setItem(storageKey, newTheme);
      setTheme(newTheme);
    },
  }

  return (
    <ThemeProviderContext.Provider {...props} value={value}>
      {children}
    </ThemeProviderContext.Provider>
  )
}

export const useTheme = () => {
  const context = useContext(ThemeProviderContext)
  if (context === undefined)
    throw new Error("useTheme must be used within a ThemeProvider")
  return context
}
