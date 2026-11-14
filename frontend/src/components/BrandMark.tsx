export function BrandMark({ small = false }: { small?: boolean }) {
  return (
    <span
      className={`brand-mark${small ? " brand-mark-small" : ""}`}
      aria-hidden="true"
    >
      <svg viewBox="0 0 24 24" fill="none">
        <path
          d="M5 19V5h14v14H5Z"
          stroke="currentColor"
          strokeWidth="1.5"
          strokeLinejoin="round"
        />
        <path
          d="M5 13h8V5m0 8h6m-6 0v6"
          stroke="currentColor"
          strokeWidth="1.5"
        />
        <path
          d="m14.5 17 1.5 1.5 3-3.5"
          stroke="currentColor"
          strokeWidth="1.5"
          strokeLinecap="round"
          strokeLinejoin="round"
        />
      </svg>
    </span>
  );
}
