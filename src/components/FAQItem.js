import React, { useState } from 'react';
import styles from '../css/FAQItem.module.css';

const FAQItem = ({ question, answer }) => {
  const [isOpen, setIsOpen] = useState(false);

  return (
    <div className={styles.faqItem}>
      <button
        className={`${styles.faqQuestion} ${isOpen ? styles.open : ''}`}
        onClick={() => setIsOpen(!isOpen)}
      >
        <span className={`${styles.icon} ${isOpen ? styles.iconOpen : ''}`} aria-hidden="true"></span>
        <span className={styles.questionText}>{question}</span>
      </button>
      {isOpen && <div className={styles.faqAnswer}>{answer}</div>}
    </div>
  );
};

export default FAQItem;
