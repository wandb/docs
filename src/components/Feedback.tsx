import React, {useCallback, useState} from 'react';

declare global {
  interface Window {
    analytics: any;
  }
}

type ClickableThumbProps = {
  handleClick: () => void;
  emoji: string;
};

const ClickableThumb: React.FC<ClickableThumbProps> = ({
  handleClick,
  emoji,
}) => {
  return (
    <span
      role="button"
      style={{padding: '0px 8px 0px 8px', cursor: 'pointer'}}
      onClick={handleClick}>
      {emoji}
    </span>
  );
};

const Feedback: React.FC = () => {
  const [feedbackPressed, setFeedbackPressed] = useState(false);
  const onThumbsUp = useCallback(() => {
    setFeedbackPressed(true);

    if (process.env.NODE_ENV === 'development') {
      console.log('Thumbs up on Docs Page:', window.location.pathname);
      return;
    }

    window.analytics.track('Thumbs Up on Docs Page', {
      page: window.location.pathname,
    });
  }, []);
  const onThumbsDown = useCallback(() => {
    setFeedbackPressed(true);

    if (process.env.NODE_ENV === 'development') {
      console.log('Thumbs down on Docs Page:', window.location.pathname);
      return;
    }

    window.analytics.track('Thumbs Down on Docs Page', {
      page: window.location.pathname,
    });
  }, []);

  const innerContent = feedbackPressed ? (
    <span>Thank you for your feedback!</span>
  ) : (
    <span>
      {' '}
      Was this page helpful?
      <ClickableThumb emoji="👍" handleClick={onThumbsUp} />
      <ClickableThumb emoji="👎" handleClick={onThumbsDown} />
    </span>
  );

  return (
    <div
      style={{
        display: 'flex',
        marginTop: '24px',
        color: 'rgba(136,153,168,1.00)',
        fontWeight: 600,
      }}>
      {innerContent}
    </div>
  );
};

export default Feedback;
