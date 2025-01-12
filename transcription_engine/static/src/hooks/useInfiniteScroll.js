// File: transcription_engine/static/src/hooks/useInfiniteScroll.js

import { useEffect, useRef, useState } from 'react';

/**
 * Custom hook for implementing infinite scroll functionality
 * @param {Array} items - Full array of items to be paginated
 * @param {number} pageSize - Number of items to load per page
 * @returns {Object} Object containing visible items and loading state
 */
export const useInfiniteScroll = (items = [], pageSize = 10) => {
  const [visibleItems, setVisibleItems] = useState([]);
  const [page, setPage] = useState(1);
  const [loading, setLoading] = useState(false);
  const [hasMore, setHasMore] = useState(true);
  const loaderRef = useRef(null);

  // Load initial items
  useEffect(() => {
    setVisibleItems(items.slice(0, pageSize));
    setHasMore(items.length > pageSize);
  }, [items, pageSize]);

  // Set up intersection observer
  useEffect(() => {
    const observer = new IntersectionObserver(
      (entries) => {
        const target = entries[0];
        if (target.isIntersecting && hasMore && !loading) {
          setLoading(true);

          // Simulate network delay for smooth loading experience
          setTimeout(() => {
            const nextItems = items.slice(0, (page + 1) * pageSize);
            setVisibleItems(nextItems);
            setPage(page + 1);
            setHasMore(nextItems.length < items.length);
            setLoading(false);
          }, 300);
        }
      },
      {
        root: null,
        rootMargin: '20px',
        threshold: 0.1,
      }
    );

    if (loaderRef.current) {
      observer.observe(loaderRef.current);
    }

    return () => {
      if (loaderRef.current) {
        observer.unobserve(loaderRef.current);
      }
    };
  }, [items, page, pageSize, hasMore, loading]);

  return {
    visibleItems,
    loading,
    hasMore,
    loaderRef,
  };
};
